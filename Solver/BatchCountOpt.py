import pyomo.environ as pyo
import pandas as pd
import time
import os
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


# 计算渠道处理时间
def get_proc_time(j, df_data, model=None):
    tot_len = 0
    tot_num = 0
    tot_stock_tm = 0  # 板数处理时间
    max_rest_stock = 4  # 每个波次中最大的散板数量
    if not model:  # 结果统计
        line_z = 'Line{}'.format(j)
        line_s = ['Line{0}-{1}-s'.format(j, idx) for idx in range(max_rest_stock)]
        for i in df_data.index:
            rest_str = [f'R{idx}' for idx in range(max_rest_stock)]
            rest_num = [df_data[rest_str_i].iloc[i] for rest_str_i in rest_str]
            tot_len += (df_data[line_z].iloc[i] * df_data['Spec'].iloc[i] + sum(df_data[line_s[idx]].iloc[i] *
                                                                                rest_num[idx] for idx in
                                                                                range(max_rest_stock))) * \
                       df_data['Length'].iloc[i]
            tot_num += df_data[line_z].iloc[i] * df_data['Spec'].iloc[i] + sum(df_data[line_s[idx]].iloc[i] *
                                                                               rest_num[idx] for idx in
                                                                               range(max_rest_stock))  # 需要修改，散货余量
            tot_stock_tm += (df_data[line_z].iloc[i] + sum(
                df_data[line_s[idx]].iloc[i] for idx in range(max_rest_stock))) * 1

    else:  # 建模生成
        for i in df_data.index:
            rest_str = [f'R{idx}' for idx in range(max_rest_stock)]
            rest_num = [df_data[rest_str_i].iloc[i] for rest_str_i in rest_str]
            rest_num_j = sum(model.u[i, j, idx] * rest_num[idx] for idx in range(max_rest_stock))
            tot_len += (model.x[i, j] * df_data['Spec'].iloc[i] + rest_num_j) * df_data['Length'].iloc[i]
            tot_num += (model.x[i, j] * df_data['Spec'].iloc[i] + rest_num_j)
            tot_stock_tm += (model.x[i, j] + sum(model.u[i, j, idx] for idx in range(max_rest_stock))) * 1
    proc_time = (tot_len + 75 * tot_num) / 4500 + tot_stock_tm
    return proc_time

# 建立数学模型
def set_model(df_data):
    model = pyo.ConcreteModel()
    # 决策变量
    line_num = df_data['Line.No'].iloc[0]  # 渠道数
    max_rest_stock = 4  # 最大散板数量
    tot_len = sum((df_data['F.No'].iloc[i] * df_data['Spec'].iloc[i] + sum(
        df_data[f'R{idx}'].iloc[i] for idx in range(max_rest_stock))) * df_data['Length'].iloc[i] for i in
                  df_data.index)
    tot_num = sum(df_data['F.No'].iloc[i] * df_data['Spec'].iloc[i] + sum(
        df_data[f'R{idx}'].iloc[i] for idx in range(max_rest_stock)) for i in df_data.index)
    tot_stock_tm = sum(df_data['F.No'].iloc[i] + df_data['R.No'].iloc[i] for i in df_data.index)
    max_proc_time = (tot_len + 75 * tot_num) / 4500 + tot_stock_tm  # 边界处理
    model.x = pyo.Var(df_data.index, range(line_num), bounds=(0, df_data['F.No'].max()),
                      domain=pyo.NonNegativeIntegers)  # 选择整版的数目
    model.u = pyo.Var(df_data.index, range(line_num), range(max_rest_stock), domain=pyo.Binary)  # 选择散版的数目
    model.y = pyo.Var(range(2), bounds=(0, max_proc_time), domain=pyo.NonNegativeReals)  # 连续变量，因为要乘以长度，物理意义为平均处理时长
    # 目标函数
    model.obj = pyo.Objective(expr=model.y[0] - model.y[1], sense=pyo.minimize)
    # 边界约束
    model.bounds = pyo.ConstraintList()
    for j in range(line_num):
        model.bounds.add(expr=model.y[0] >= get_proc_time(j, df_data, model))  # 计算每个渠道总的处理时长
        model.bounds.add(expr=model.y[1] <= get_proc_time(j, df_data, model))
    # 板量约束（版型约束）
    model.stocks = pyo.ConstraintList()
    # 货量约束
    model.counts = pyo.ConstraintList()
    for i in df_data.index:
        rest_str = [f'R{idx}' for idx in range(max_rest_stock)]
        rest_num = [df_data[rest_str_i].iloc[i] for rest_str_i in rest_str]  # 需要保证散版的数量
        model.stocks.add(
            expr=sum(model.x[i, j] for j in range(line_num)) == df_data['F.No'].iloc[i])  # 整版的数量也固定
        model.stocks.add(expr=sum(model.u[i, j, idx] for idx in range(max_rest_stock) for j in range(line_num)) ==
                              df_data['R.No'].iloc[i])  # 散版的数量固定
        for idx in range(max_rest_stock):
            model.stocks.add(expr=sum(model.u[i, j, idx] for j in range(line_num)) <= 1)  # 散板只能选择一个渠道进行投放
        model.counts.add(
            expr=sum(model.x[i, j] * df_data['Spec'].iloc[i] + sum(
                model.u[i, j, idx] * rest_num[idx] for idx in range(max_rest_stock)) for j in range(line_num)) ==
                 df_data['R.Count'].iloc[i] + df_data['F.No'].iloc[i] * df_data['Spec'].iloc[i])

    # model.pprint()
    return model

# 求解器设置
def cal_obj(model, mtype):
    print("----Solving----")
    if mtype == 'nlp':
        bonmin_path = './CoinAll-1.6.0-win64-intel11.1/CoinAll-1.6.0-win64-intel11.1/bin/bonmin.exe'
        opt = pyo.SolverFactory('bonmin', executable=bonmin_path)  # 求解器的选择
        opt.options['bonmin.time_limit'] = 120  # bonmin求解时间限制，单位秒
    elif mtype == 'lp':
        glpk_path = './solver/winglpk-4.65/glpk-4.65/w64/glpsol.exe'
        opt = pyo.SolverFactory('glpk', executable=glpk_path)  # 求解器的选择
        opt.options['tmlim'] = 120  # glpk求解时间限制，单位秒
    results = opt.solve(model, tee=False)
    results.write()

# 调用多线程计算
def get_multi_task(df_wave_data, max_rest_stock, max_line_num, col, proc_col):
    wave = df_wave_data['Wave'].iloc[0]
    print("Wave {0} start! Port num is {1}\n".format(wave, os.getpid()))
    line_num = df_wave_data['Line.No'].iloc[0]  # 管道数
    model = set_model(df_wave_data)  # 实例化模型
    cal_obj(model, 'nlp')  # lp是线性求解器，nlp是非线性求解器
    print('Wave {} Complete!'.format(wave))
    print('Min cost:', model.obj())
    out = []
    for i in df_wave_data.index:
        out_i = [wave, df_wave_data['Code'].iloc[i]]
        for j in range(line_num):  # 渠道数
            s_zero_one = [pyo.value(model.u[i, j, idx]) for idx in range(max_rest_stock)]
            out_i += [pyo.value(model.x[i, j])] + s_zero_one
        if line_num < max_line_num:
            delta = max_line_num - line_num
            out_i += [0] * delta * (1 + max_rest_stock)
        out.append(out_i)
    wave_out = pd.DataFrame(out, columns=col)
    res = pd.merge(df_wave_data, wave_out, on='Code')
    # 各渠道的处理时间
    proc_time = [wave]
    for j in range(line_num):
        proc_time.append(get_proc_time(j, res))
    if line_num < max_line_num:
        delta = max_line_num - line_num
        proc_time += [0] * delta
    procTm = pd.DataFrame([proc_time], columns=proc_col)
    return {"result": wave_out, "procTm": procTm}


if __name__ == "__main__":
    df_data = pd.read_excel('./data/batch_data.xlsx')  # 输入数据
    max_rest_stock = df_data['R.No'].max()  # 全局的最大散版数量
    max_line_num = df_data['Line.No'].max()  # 全局的最大管道数
    # 生成结果输出表   
    col = ['Wave', 'Code']  # 索引字段
    proc_col = ['Wave']
    for j in range(max_line_num):
        line_z = ['Line{}'.format(j)]
        line_s = ['Line{0}-{1}-s'.format(j, idx) for idx in range(max_rest_stock)]
        col += line_z + line_s
        proc_col += line_z
    out = pd.DataFrame(columns=col)
    proc_res = pd.DataFrame(columns=proc_col)
    wave_list = df_data['Wave'].drop_duplicates().tolist()  # 波次去重后生成list
    # 按波次提取
    tm1 = time.time()
    pool = Pool(10)  # 调用多进程求解
    result = []
    for wave in wave_list:
        df_wave_data = df_data[df_data['Wave'] == wave].reset_index(drop=True)  # 提取波次数据
        res = pool.apply_async(get_multi_task, (df_wave_data, max_rest_stock, max_line_num, col, proc_col,))
        result.append(res)
    pool.close()
    pool.join()  # 在进程中可以阻塞主进程的执行, 直到等待子线程全部完成之后, 才继续运行主线程后面的代码
    # 结果处理
    for res in result:
        res_dict = res.get()
        proc_res = proc_res.append(res_dict['procTm'])
        out = out.append(res_dict['result'])

    print("所有波次求解总时长：", time.time() - tm1)
    out.to_excel('./data/result_num_new.xlsx', index=False)  # 输出结果
    proc_res.to_excel('./data/result_proc_time_new.xlsx', index=False)  # 输出结果
