# -*- coding: utf-8 -*-
"""方案A日前计划：在同一个优化模型中执行逐笔120元价差约束。"""
import numpy as np
import pulp

import 寻优 as M


EPS = 1e-7


def _values(variables, n):
    return np.array([float(variables[t].value() or 0.0) for t in range(n)])


def solve_window_a(P, H, init_soc, curt, fc, price, tou, need,
                   end_frac=M.SOC_MIN, P_STATION=None,
                   return_pairs=False, ramp=None,
                   initial_inventory=None):
    """求解一天的方案A日前计划。

    付费充电按预测电价计成本，并直接与当天后续放电建立电量配对；只有
    预测放电价减预测充电价不低于120元/兆瓦时的配对才进入模型。免费
    充电按零成本记账，预测放电价值不低于120元/兆瓦时才允许放电；
    日初已有电量保留各自的最低预测放电价格，不跨日丢失成本来源。
    """
    curt = np.asarray(curt, dtype=float)
    forecast = np.asarray(fc, dtype=float)
    forecast_price = np.asarray(price, dtype=float)
    utility_price = np.asarray(tou, dtype=float)
    grid_need = np.asarray(need, dtype=float)
    n = len(forecast_price)
    points = range(n)

    capacity = P * H
    minimum_energy = M.SOC_MIN * capacity
    maximum_energy = M.SOC_MAX * capacity
    end_energy = end_frac * capacity
    initial_energy = max(0.0, float(init_soc) - minimum_energy)
    if initial_inventory is None:
        initial_lots = ([(initial_energy, M.ARB_MIN)]
                        if initial_energy > EPS else [])
    else:
        initial_lots = [
            (float(amount), float(minimum_discharge_price))
            for amount, minimum_discharge_price in initial_inventory
            if float(amount) > EPS
        ]
        supplied_energy = sum(amount for amount, _ in initial_lots)
        if abs(supplied_energy - initial_energy) > 1e-5:
            raise ValueError(
                "日初库存分账电量与实际SOC不一致:"
                f"分账={supplied_energy:.6f}MWh,实际={initial_energy:.6f}MWh")

    problem = pulp.LpProblem("scheme_a_day_ahead", pulp.LpMaximize)
    free_charge = pulp.LpVariable.dicts("free_charge", points, 0)
    paid_charge = pulp.LpVariable.dicts("paid_charge", points, 0)
    utility_discharge = pulp.LpVariable.dicts("utility_discharge", points, 0)
    market_discharge = pulp.LpVariable.dicts("market_discharge", points, 0)
    energy = pulp.LpVariable.dicts(
        "energy", points, minimum_energy, maximum_energy)
    free_inventory = pulp.LpVariable.dicts(
        "free_inventory", points, 0, maximum_energy - minimum_energy)
    free_to_utility = pulp.LpVariable.dicts("free_to_utility", points, 0)
    free_to_market = pulp.LpVariable.dicts("free_to_market", points, 0)
    charge_mode = pulp.LpVariable.dicts("charge_mode", points, cat="Binary")

    charge_points = [
        i for i in points
        if forecast[i] > EPS and forecast_price[i] <= M.CHG_PRICE_MAX
    ]
    market_pairs = [
        (i, j) for i in charge_points for j in range(i + 1, n)
        if forecast_price[j] - forecast_price[i] >= M.ARB_MIN - EPS
    ]
    utility_pairs = [
        (i, j) for i in charge_points for j in range(i + 1, n)
        if grid_need[j] > EPS
        and utility_price[j] - forecast_price[i] >= M.ARB_MIN - EPS
    ]
    paid_to_market = pulp.LpVariable.dicts(
        "paid_to_market", market_pairs, 0)
    paid_to_utility = pulp.LpVariable.dicts(
        "paid_to_utility", utility_pairs, 0)

    # 跨日库存从当天开始即已存在。每批库存保留其最低预测放电价格：
    # 免费库存为120元/MWh；付费库存为原预测充电价+120元/MWh。
    initial_to_market = {}
    initial_to_utility = {}
    initial_market_in = {j: [] for j in points}
    initial_utility_in = {j: [] for j in points}
    for lot, (amount, minimum_discharge_price) in enumerate(initial_lots):
        lot_vars = []
        for j in points:
            if forecast_price[j] >= minimum_discharge_price - EPS:
                variable = pulp.LpVariable(
                    f"initial_to_market_{lot}_{j}", lowBound=0)
                initial_to_market[lot, j] = variable
                initial_market_in[j].append(variable)
                lot_vars.append(variable)
            if (grid_need[j] > EPS
                    and utility_price[j] >= minimum_discharge_price - EPS):
                variable = pulp.LpVariable(
                    f"initial_to_utility_{lot}_{j}", lowBound=0)
                initial_to_utility[lot, j] = variable
                initial_utility_in[j].append(variable)
                lot_vars.append(variable)
        problem += pulp.lpSum(lot_vars) <= amount

    market_in = {j: [] for j in points}
    utility_in = {j: [] for j in points}
    paid_out = {i: [] for i in points}
    for pair in market_pairs:
        paid_to_market_vars = paid_to_market[pair]
        paid_out[pair[0]].append(paid_to_market_vars)
        market_in[pair[1]].append(paid_to_market_vars)
    for pair in utility_pairs:
        paid_to_utility_vars = paid_to_utility[pair]
        paid_out[pair[0]].append(paid_to_utility_vars)
        utility_in[pair[1]].append(paid_to_utility_vars)

    for t in points:
        problem += free_charge[t] <= max(0.0, curt[t])
        paid_limit = max(0.0, forecast[t]) if paid_out[t] else 0.0
        problem += paid_charge[t] <= paid_limit
        problem += free_charge[t] + paid_charge[t] <= P * charge_mode[t]
        problem += utility_discharge[t] + market_discharge[t] <= P * (1 - charge_mode[t])
        problem += utility_discharge[t] <= max(0.0, grid_need[t])

        # 限电充电等免费电量按0元/MWh记账，也必须满足120元/MWh
        # 的预测价差；低于门槛时不得从免费库存放电。
        if utility_price[t] < M.ARB_MIN - EPS:
            problem += free_to_utility[t] == 0
        if forecast_price[t] < M.ARB_MIN - EPS:
            problem += free_to_market[t] == 0

        # 放电的储能侧电量必须来自免费库存或一笔满足120元价差的付费充电。
        problem += utility_discharge[t] * M.DT / M.ETA == (
            free_to_utility[t] + pulp.lpSum(utility_in[t])
            + pulp.lpSum(initial_utility_in[t]))
        problem += market_discharge[t] * M.DT / M.ETA == (
            free_to_market[t] + pulp.lpSum(market_in[t])
            + pulp.lpSum(initial_market_in[t]))
        # 当天新安排的付费充电必须在当天后续完成合格配对，不结转为免费库存。
        problem += paid_charge[t] * M.DT == pulp.lpSum(paid_out[t])

        if t == 0:
            problem += energy[t] == (
                init_soc + (free_charge[t] + paid_charge[t]) * M.DT
                - (utility_discharge[t] + market_discharge[t]) * M.DT / M.ETA)
            problem += free_inventory[t] == (
                free_charge[t] * M.DT
                - free_to_utility[t] - free_to_market[t])
        else:
            problem += energy[t] == (
                energy[t - 1] + (free_charge[t] + paid_charge[t]) * M.DT
                - (utility_discharge[t] + market_discharge[t]) * M.DT / M.ETA)
            problem += free_inventory[t] == (
                free_inventory[t - 1] + free_charge[t] * M.DT
                - free_to_utility[t] - free_to_market[t])
        problem += free_inventory[t] <= energy[t] - minimum_energy

    problem += energy[n - 1] >= end_energy

    if ramp is not None:
        for t in range(1, n):
            station_power = (
                forecast[t] - free_charge[t] - paid_charge[t]
                + utility_discharge[t] + market_discharge[t])
            previous_power = (
                forecast[t - 1] - free_charge[t - 1] - paid_charge[t - 1]
                + utility_discharge[t - 1] + market_discharge[t - 1])
            problem += station_power - previous_power <= ramp
            problem += previous_power - station_power <= ramp

    problem += pulp.lpSum(
        M.DT * (
            utility_price[t] * utility_discharge[t]
            + forecast_price[t] * market_discharge[t]
            - forecast_price[t] * paid_charge[t]
        ) for t in points)

    status = problem.solve(pulp.HiGHS(
        msg=False, gapRel=1e-5, timeLimit=60, threads=1))
    if pulp.LpStatus[status] not in {"Optimal", "Integer Feasible"}:
        # 个别大容量日期在并行网格中60秒内无法形成可用整数解时，
        # 仅放宽最优性证明精度并延长求解时间；全部调度硬约束保持不变。
        status = problem.solve(pulp.HiGHS(
            msg=False, gapRel=1e-4, timeLimit=180, threads=1))
    if pulp.LpStatus[status] not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(
            f"方案A日前计划求解失败:{pulp.LpStatus[status]}")

    result = (
        _values(free_charge, n),
        _values(paid_charge, n),
        _values(utility_discharge, n),
        _values(market_discharge, n),
    )
    if not return_pairs:
        return result

    audit = []
    for (i, j), variable in paid_to_utility.items():
        amount = float(variable.value() or 0.0)
        if amount > EPS:
            audit.append((i, j, 0, amount, forecast_price[i]))
    for (i, j), variable in paid_to_market.items():
        amount = float(variable.value() or 0.0)
        if amount > EPS:
            audit.append((i, j, 1, amount, forecast_price[i]))
    audit.sort(key=lambda item: (item[0], item[1], item[2]))
    return result, audit
