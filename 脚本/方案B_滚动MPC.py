# -*- coding: utf-8 -*-
"""方案 A/B 的滚动仿真:决策按各自信息结构、结算用实际。

三方案对应:
- 方案 A(当前现实)   = solve_mpc : 日前申报、日前出清、运行日计划锁死不能改。
   每天解一个 96 点窗口(日前预测视角 forecast_mw/mkt_fc),只执行当天;
   实际 SOC 用实际发电/价格推进(被动微调),全年 365 次滚动。
- 方案 B(以后规则)   = solve_mpc_b: 运行日每小时滚动申报「下一小时→24:00」、省调滚动出清。
   滚动出清未来 2h 是现行机制(不受 24 点限制):电价用出清价(≈实际 mkt);
   场站出力仅负电价(限电)时段由 AGC 指令透露(act≈agc),正电价时段仍用日前预测。
- 方案 C(理想上界)   = 寻优.solve_year: 全年耦合全知。

共同口径(用户拍板):
- 弃电充电始终用实际 curt(下界对照);末端 SOC 回 5%(=SOC_MIN,纯日前口径:窗口内放空、无跨日结转预留);
- 充放价差门槛:方案A付费充电逐笔满足预测放电价−预测充电价≥120;免费电量按0元成本,预测放电价值≥120才允许;
- 充电电价上限:发电充电 cg 仅在电价 ≤250 元/MWh 时允许(弃电充电 cc 仅在 mkt≤−80 时发生,天然满足);
- 结算用实际 act/mkt(决策≠结算)。
"""
import sys
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "脚本")
import 寻优 as M
from 方案A_日前调度 import solve_window_a

ND = 96              # 每日点数
E_REF = 0.05         # 窗口末端 SOC 参考值 = SOC_MIN(纯日前口径:窗口内放空、无跨日结转预留)
ARB_MIN = M.ARB_MIN  # 方案B窗口的价差开关；方案A使用 solve_window_a 的逐笔价差约束


def _inventory_energy(inventory):
    return float(sum(inventory.values()))


def _add_inventory(inventory, amount, minimum_discharge_price):
    """增加一批储能侧电量，并保留其最低预测放电价格。"""
    if amount <= 1e-9:
        return
    key = round(float(minimum_discharge_price), 8)
    inventory[key] = inventory.get(key, 0.0) + float(amount)


def _take_inventory(inventory, requested_energy, discharge_value):
    """从满足价差的库存中取电；优先消耗门槛最高的批次。"""
    remaining = max(0.0, float(requested_energy))
    taken = 0.0
    eligible = sorted(
        (key for key, amount in inventory.items()
         if amount > 1e-9 and key <= discharge_value + 1e-7),
        reverse=True)
    for key in eligible:
        amount = min(remaining, inventory[key])
        inventory[key] -= amount
        remaining -= amount
        taken += amount
        if inventory[key] <= 1e-9:
            del inventory[key]
        if remaining <= 1e-9:
            break
    return taken


def solve_window(P, H, init_soc, curt, fc, mkt_fc, tou, gn, end_frac=E_REF, end_ref=True, P_STATION=None, ramp=None):
    """解一个 96 点窗口 MILP(决策视角)。

    发电充电上限 = 预测发电 fc;弃电充电上限 = 实际弃电 curt;
    价格 = 预测 mkt_fc;末端 SOC >= end_frac(极端无电窗口回退到无末端约束)。
    充放价差门槛(开关):发电充电 cg 成本用真实预测价,仅当「未来窗口最高价 − 当前预测价 ≥ ARB_MIN」才允许充电。
    返回 (cc, cg, pu, pv) 计划序列。
    """
    Ps = M.P_STATION if P_STATION is None else P_STATION
    E_cap = P * H
    E_lo = M.SOC_MIN * E_cap
    E_hi = M.SOC_MAX * E_cap
    E_ref = end_frac * E_cap
    N = len(curt)
    idx = range(N)
    mkt_fc_arr = np.asarray(mkt_fc, dtype=float)
    future_max = np.full(N, -1e9)   # future_max[t] = t 之后(不含)窗口内最高预测价
    for t in range(N - 2, -1, -1):
        future_max[t] = max(future_max[t + 1], mkt_fc_arr[t + 1])
    prob = pulp.LpProblem("win", pulp.LpMaximize)
    cc = pulp.LpVariable.dicts("c_c", idx, 0)
    cg = pulp.LpVariable.dicts("c_g", idx, 0)
    pu = pulp.LpVariable.dicts("u", idx, 0)
    pv = pulp.LpVariable.dicts("v", idx, 0)
    E = pulp.LpVariable.dicts("E", idx, E_lo, E_hi)
    for t in idx:
        can_arb = (future_max[t] - mkt_fc_arr[t]) >= ARB_MIN and mkt_fc_arr[t] <= M.CHG_PRICE_MAX
        prob += cc[t] <= curt[t]
        prob += cg[t] <= (max(0.0, fc[t]) if can_arb else 0.0)
        prob += cc[t] + cg[t] <= P
        prob += pu[t] + pv[t] <= P
        prob += pu[t] <= gn[t]
    z = pulp.LpVariable.dicts("z", idx, cat="Binary")
    for t in idx:
        prob += cc[t] + cg[t] <= P * z[t]
        prob += pu[t] + pv[t] <= P * (1 - z[t])
    prob += E[0] == init_soc + (cc[0] + cg[0]) * M.DT - (pu[0] + pv[0]) / M.ETA * M.DT
    for t in range(1, N):
        prob += E[t] == E[t - 1] + (cc[t] + cg[t]) * M.DT - (pu[t] + pv[t]) / M.ETA * M.DT
    if end_ref:
        prob += E[N - 1] >= E_ref
    # 场站有功功率爬坡约束(15min 相邻变化 ≤ ramp):场站总功率 = 预测发电 − 充电 + 放电
    if ramp is not None:
        for t in range(1, N):
            ch = cc[t] + cg[t]; dis = pu[t] + pv[t]
            ch_p = cc[t - 1] + cg[t - 1]; dis_p = pu[t - 1] + pv[t - 1]
            g = float(fc[t]) - ch + dis
            gp = float(fc[t - 1]) - ch_p + dis_p
            prob += g - gp <= ramp
            prob += gp - g <= ramp
    # 充放价差门槛(开关):发电充电 cg 成本用真实预测价,仅当「未来窗口最高价 − 当前价 ≥ ARB_MIN」才允许(见上 can_arb)。
    # 弃电充电 cc 成本 0(免费)、抵厂用电 pu 收入 tou(固定高价)不受影响。
    prob += pulp.lpSum(M.DT * (tou[t] * pu[t] + mkt_fc_arr[t] * pv[t] - mkt_fc_arr[t] * cg[t]) for t in idx)
    prob.solve(pulp.HiGHS(msg=False, gapRel=0.01, timeLimit=30, threads=1))
    if pulp.value(prob.objective) is None:
        if end_ref:
            # 极端窗口(无电可充):放松末端约束,允许放空
            return solve_window(P, H, init_soc, curt, fc, mkt_fc, tou, gn, end_frac, end_ref=False, P_STATION=P_STATION)
        return np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    cc_ = np.array([cc[t].value() for t in idx])
    cg_ = np.array([cg[t].value() for t in idx])
    pu_ = np.array([pu[t].value() for t in idx])
    pv_ = np.array([pv[t].value() for t in idx])
    return cc_, cg_, pu_, pv_


def solve_mpc(P, H, df, verbose=False, end_frac=E_REF, return_schedule=False, P_STATION=None, ramp=None):
    """滚动 MPC 全年仿真,返回 (R_year, r_cap, (rev_tou, rev_mkt, cost_chg))。

    return_schedule=True 时额外返回第 4 元:schedule dict(cc/cg/pu/pv/soc 全年执行序列)。
    """
    E_cap = P * H
    E_lo = M.SOC_MIN * E_cap
    E_hi = M.SOC_MAX * E_cap
    act = df["act_mw"].values
    avail = df["avail"].values
    curt = df["curt"].values
    fc = df["forecast_mw"].values
    mkt = df["mkt"].values
    mkt_fc = df["mkt_fc"].values
    tou = df["tou"].values
    gn = df["grid_need"].values
    N = len(df)
    n_days = N // ND

    cc_exec = np.zeros(N)
    cg_exec = np.zeros(N)
    pu_exec = np.zeros(N)
    pv_exec = np.zeros(N)
    soc_exec = np.zeros(N)

    # 实际库存按最低预测放电价格分账，避免付费电量跨日后被误作免费电量。
    # 键=最低预测放电价格，值=储能侧电量(MWh)；5%最低SOC不参与调度。
    inventory = {}
    Ps = M.P_STATION if P_STATION is None else P_STATION  # 新能源装机(执行放电约束用)
    for d in range(n_days):
        t0 = d * ND
        s = slice(t0, t0 + ND)
        soc = E_lo + _inventory_energy(inventory)
        try:
            cc_p, cg_p, pu_p, pv_p = solve_window_a(
                P, H, soc, curt[s], fc[s], mkt_fc[s], tou[s], gn[s],
                end_frac, P_STATION=P_STATION, ramp=ramp,
                initial_inventory=[(amount, price_floor)
                                   for price_floor, amount in inventory.items()])
        except RuntimeError as exc:
            date = df["time"].iloc[t0].date()
            raise RuntimeError(
                f"方案A求解失败:P={P},H={H},日期={date};{exc}") from exc
        gap_E = 0.0   # 当天累积缺口(电量):发电充电少充的量,当天清零
        for k in range(ND):
            t = t0 + k
            soc = E_lo + _inventory_energy(inventory)
            charge_cap = min(P, (E_hi - soc) / M.DT)

            # ① 日前计划优先执行：免费充电受实际弃电限制，付费充电受实际发电限制。
            cc = min(cc_p[k], max(0.0, curt[t]), charge_cap)
            cg = min(cg_p[k], max(0.0, act[t]), charge_cap - cc)

            # ② 仅用此前累积缺口进行补充电，不挤占本时点的日前计划；
            #    补充电取自「可用−实际」，免费，且计划放电时禁止补充电。
            surplus = max(0.0, avail[t] - act[t])
            remaining_charge_cap = max(0.0, charge_cap - cc - cg)
            supp = 0.0 if pu_p[k] + pv_p[k] > 1e-7 else min(
                gap_E,
                min(remaining_charge_cap,
                    max(0.0, surplus - cc)) * M.DT)
            cc = cc + supp / M.DT
            gap_E -= supp

            # ③ 本时点付费充电未执行部分形成新缺口，只能由后续时点弥补。
            gap_E += max(0.0, cg_p[k] - cg) * M.DT
            # ④ 将实际充电按来源入账：免费电量门槛为120；付费电量门槛为
            #    原预测充电价+120。跨日时继续保留该门槛。
            _add_inventory(inventory, cc * M.DT, M.ARB_MIN)
            _add_inventory(
                inventory, cg * M.DT, mkt_fc[t] + M.ARB_MIN)

            # ⑤ 按日前计划放电。实际库存必须满足对应预测放电价值；
            #    抵厂用电优先，套利放电再受「新能源功率+放电功率≤装机」限制。
            pu_plan_E = pu_p[k] / M.ETA * M.DT
            pv_power = min(pv_p[k], max(0.0, Ps - act[t]))
            pv_plan_E = pv_power / M.ETA * M.DT
            pu_E = _take_inventory(inventory, pu_plan_E, tou[t])
            pv_E = _take_inventory(inventory, pv_plan_E, mkt_fc[t])
            soc = E_lo + _inventory_energy(inventory)
            cc_exec[t] = cc
            cg_exec[t] = cg
            pu_exec[t] = pu_E / M.DT * M.ETA
            pv_exec[t] = pv_E / M.DT * M.ETA
            soc_exec[t] = soc
        if verbose and (d + 1) % 60 == 0:
            print(f"  已滚动 {d + 1}/{n_days} 天, 当前实际 SOC={soc / E_cap:.2%}", flush=True)

    # 结算:免费充电 = 弃电充电 + 补充电(都当弃电,免费、不吞上网电量);付费充电 = 发电充电 cg
    pc_exec = cc_exec + cg_exec
    pd_exec = pu_exec + pv_exec
    pc_gen_settle = cg_exec   # 发电充电(付费、吞掉本应上网的电,容量补偿扣减)
    u = np.minimum(pd_exec, gn)
    rev_tou = M.DT * float((tou * u).sum())
    rev_mkt = M.DT * float((mkt * (pd_exec - u)).sum())
    cost_chg = M.DT * float((mkt * pc_gen_settle).sum())
    R_year = rev_tou + rev_mkt - cost_chg
    r_cap = M._capacity_comp(df, pd_exec, pc_gen_settle)
    if return_schedule:
        sched = dict(cc=cc_exec, cg=cg_exec, pu=pu_exec, pv=pv_exec, soc=soc_exec)
        return R_year, r_cap, (rev_tou, rev_mkt, cost_chg), sched
    return R_year, r_cap, (rev_tou, rev_mkt, cost_chg)


AHEAD = 8  # 未来 2 h 准确(8 个 15min 点)


def solve_mpc_b(P, H, df, verbose=False, end_frac=E_REF, return_schedule=False, ramp=None):
    """方案 B(运行日滚动申报):每小时滚动报「下一小时→24:00」的充放电计划,省调滚动出清。

    信息结构(用户口径):
    - 滚动出清未来 2h 是现行机制(非未来),且不受 24 点限制(连续 2h 窗口)。
      它给出未来 2h 的准信息:电价=出清价(≈实际 mkt)、实际出力 act、可用 avail;
      弃电 curt = avail − act 直接可知(实际小于可用即限电),无需 AGC 判断限电。
    - 2h 之后到日末:电价用日前预测 mkt_fc、出力用日前预测 forecast。
    - 弃电充电始终用实际 curt(下界口径,用户拍板)。末端 SOC 回 end_frac。
    每小时解一次窗口(24 次/天),执行未来 1h(4 点)。
    """
    E_cap = P * H
    E_lo = M.SOC_MIN * E_cap
    E_hi = M.SOC_MAX * E_cap
    act = df["act_mw"].values
    curt = df["curt"].values
    fc = df["forecast_mw"].values
    mkt = df["mkt"].values
    mkt_fc = df["mkt_fc"].values
    tou = df["tou"].values
    gn = df["grid_need"].values
    N = len(df)
    n_days = N // ND

    cc_exec = np.zeros(N)
    cg_exec = np.zeros(N)
    pu_exec = np.zeros(N)
    pv_exec = np.zeros(N)
    soc_exec = np.zeros(N)

    soc = E_lo
    for d in range(n_days):
        day_end = (d + 1) * ND
        for h in range(24):
            t0 = d * ND + h * 4
            t_end = day_end
            s = slice(t0, t_end)
            n_win = t_end - t0
            # 滚动出清未来 2h:全知实际(电价 mkt、实际出力 act;弃电 curt=avail−act 直接可知)
            in_ahead = np.arange(n_win) < AHEAD
            mkt_view = np.where(in_ahead, mkt[s], mkt_fc[s])
            act_view = np.where(in_ahead, act[s], fc[s])
            cc_p, cg_p, pu_p, pv_p = solve_window(P, H, soc, curt[s], act_view, mkt_view, tou[s], gn[s], end_frac, ramp=ramp)
            # 执行未来 1h(4 点)
            for k in range(4):
                t = t0 + k
                cc = cc_p[k]
                cg = min(cg_p[k], max(0.0, act[t])) if mkt[t] <= M.CHG_PRICE_MAX else 0.0
                charge_E = (cc + cg) * M.DT
                pu_plan_E = pu_p[k] / M.ETA * M.DT
                pv_plan_E = pv_p[k] / M.ETA * M.DT
                avail_E = max(0.0, soc + charge_E - E_lo)
                pu_E = min(pu_plan_E, avail_E)
                pv_E = min(pv_plan_E, avail_E - pu_E)
                soc = min(soc + charge_E - pu_E - pv_E, E_hi)
                soc = max(soc, E_lo)
                cc_exec[t] = cc
                cg_exec[t] = cg
                pu_exec[t] = pu_E / M.DT * M.ETA
                pv_exec[t] = pv_E / M.DT * M.ETA
                soc_exec[t] = soc
        if verbose and (d + 1) % 60 == 0:
            print(f"  已滚动 {d + 1}/{n_days} 天, 当前实际 SOC={soc / E_cap:.2%}", flush=True)

    pc_exec = cc_exec + cg_exec
    pd_exec = pu_exec + pv_exec
    pc_gen_settle = np.maximum(pc_exec - curt, 0.0)
    u = np.minimum(pd_exec, gn)
    rev_tou = M.DT * float((tou * u).sum())
    rev_mkt = M.DT * float((mkt * (pd_exec - u)).sum())
    cost_chg = M.DT * float((mkt * pc_gen_settle).sum())
    R_year = rev_tou + rev_mkt - cost_chg
    r_cap = M._capacity_comp(df, pd_exec, pc_gen_settle)
    if return_schedule:
        sched = dict(cc=cc_exec, cg=cg_exec, pu=pu_exec, pv=pv_exec, soc=soc_exec)
        return R_year, r_cap, (rev_tou, rev_mkt, cost_chg), sched
    return R_year, r_cap, (rev_tou, rev_mkt, cost_chg)


if __name__ == "__main__":
    df = M.load_data()
    P, H = 260.0, 6
    print(f"数据: {len(df)} 点, {df['time'].dt.date.nunique()} 天")

    # 方案 A:全知上界
    R_year_A, r_cap_A, comp_A = M.solve_year((P, H), df)

    # 方案 B:滚动 MPC(无天眼)
    print(f"\n=== 方案 B 滚动 MPC (P={P:.0f}, H={H}) ===", flush=True)
    R_year_B, r_cap_B, comp_B = solve_mpc(P, H, df, verbose=True)

    fA = M.project_finance(P, H, R_year_A, r_cap_A)
    fB = M.project_finance(P, H, R_year_B, r_cap_B)

    print("\n=== 方案 A vs 方案 B 对比 ===")
    print(f"  R_year    A={R_year_A/1e4:8.1f}万  B={R_year_B/1e4:8.1f}万  代价={ (R_year_A-R_year_B)/1e4:8.1f}万")
    print(f"  容量补偿  A={r_cap_A/1e4:8.1f}万  B={r_cap_B/1e4:8.1f}万  代价={ (r_cap_A-r_cap_B)/1e4:8.1f}万")
    print(f"  全投资NPV A={fA['fi_npv']/1e4:8.1f}万  B={fB['fi_npv']/1e4:8.1f}万  IRR_B={fB['fi_irr']*100:5.2f}%")
    print(f"  资本金NPV A={fA['eq_npv']/1e4:8.1f}万  B={fB['eq_npv']/1e4:8.1f}万  IRR_B={fB['eq_irr']*100:5.2f}%")
    print(f"\n  收益构成(B): 抵厂用电={comp_B[0]/1e4:.1f}万 套利={comp_B[1]/1e4:.1f}万 充电成本={comp_B[2]/1e4:.1f}万")
    print(f"  收益构成(A): 抵厂用电={comp_A[0]/1e4:.1f}万 套利={comp_A[1]/1e4:.1f}万 充电成本={comp_A[2]/1e4:.1f}万")
