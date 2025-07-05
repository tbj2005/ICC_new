import numpy as np
import bvn
import random
import tensorflow as tf
import os
import time
from collections import deque
import copy
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = time.time()


# 转对称
def make_symmetric(matrix):
    symmetric_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            symmetric_matrix[i][j] = max(matrix[i][j], matrix[j][i])

    return symmetric_matrix


def stuffing_min(matrix):
    copied_matrix = np.copy(matrix)
    while True:
        row_sum = copied_matrix.sum(axis=1)
        column_sum = copied_matrix.sum(axis=0)
        if np.array_equal(row_sum, column_sum):
            if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
                break
        max_value = max(np.maximum(row_sum, column_sum))
        flag_column = False  #寻找列为假
        row_differ = np.array([max_value] * R_S) - row_sum  #获取行的差
        column_differ = np.array([max_value] * R_S) - column_sum  #获取列的差

        s_R_d_index = np.argsort(row_differ)  #行差排序后的索引
        sort_row_differ = row_differ[s_R_d_index]  #行差排序后的结果

        s_C_d_index = np.argsort(column_differ)  #列差排序后的索引
        sort_column_differ = column_differ[s_C_d_index]  #列差排序后的结果

        # 合并排序后的行差和列差为一个新的数组
        combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)

        # 找到新数组非零元素的索引
        nonzero_indices = np.nonzero(combined_array)

        # 获取新数组非零元素
        nonzero_elements = combined_array[nonzero_indices]

        # 找到新数组非零最小值的索引和非零元素的最小值
        min_nonzero_index = np.argmin(nonzero_elements)
        min_nonzero_d = nonzero_elements[min_nonzero_index]  # 这个值就是用来填充的值

        # 返回这个数在原矩阵中的索引
        min_d_index = nonzero_indices[1][min_nonzero_index]
        # max_d_index = nonzero_indices[1][max_nonzero_index]

        # if max_d_index < R_S or min_d_index < R_S:
        if min_d_index < R_S:
            # max_p = s_R_d_index[max_d_index]
            min_p = s_R_d_index[min_d_index]  # 最小差距是行
            flag_column = True  # 此时要寻找最大差距的列
        else:
            min_p = s_C_d_index[min_d_index % R_S]  # 最小车距是列，此时要寻找最大差距的行
            # max_p = s_C_d_index[max_d_index % R_S]

        if flag_column:  # 如果要寻找最大差距的列
            max_to_stuff_index = np.argmax(column_differ)  # 获取差最大值的索引
            if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:
                copied_matrix[min_p][max_to_stuff_index] += min_nonzero_d
            else:
                if min_p != max_to_stuff_index:
                    copied_matrix[min_p][max_to_stuff_index] += min_nonzero_d
                else:
                    column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                    # 找到第二大值的索引
                    second_largest_index = np.argmax(column_differ)
                    copied_matrix[min_p][second_largest_index] += min_nonzero_d
            # 根据索引信息找到在原数组中的索引
        else:
            max_to_stuff_index = np.argmax(row_differ)  # 获取差最大值的索引
            if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:
                copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
            else:
                if min_p != max_to_stuff_index:
                    copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
                else:
                    row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                    # 找到第二大值的索引
                    second_largest_index = np.argmax(row_differ)
                    copied_matrix[second_largest_index][min_p] += min_nonzero_d
    return copied_matrix


def stuffing_new_min(matrix):
    copied_matrix = np.copy(matrix)
    while True:
        row_sum = copied_matrix.sum(axis=1)
        column_sum = copied_matrix.sum(axis=0)
        if np.array_equal(row_sum, column_sum):
            if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
                break
        max_value = max(np.maximum(row_sum, column_sum))
        flag_column = False  # 寻找列为假
        row_differ = np.array([max_value] * R_S) - row_sum  # 获取行的差
        column_differ = np.array([max_value] * R_S) - column_sum  # 获取列的差

        s_R_d_index = np.argsort(row_differ)  # 行差排序后的索引
        sort_row_differ = row_differ[s_R_d_index]  # 行差排序后的结果

        s_C_d_index = np.argsort(column_differ)  # 列差排序后的索引
        sort_column_differ = column_differ[s_C_d_index]  # 列差排序后的结果

        # 合并排序后的行差和列差为一个新的数组
        combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)

        # 找到新数组非零元素的索引
        nonzero_indices = np.nonzero(combined_array)

        # 获取新数组非零元素
        nonzero_elements = combined_array[nonzero_indices]

        # 找到新数组非零最小值的索引和非零元素的最小值
        min_nonzero_index = np.argmin(nonzero_elements)
        min_nonzero_d = nonzero_elements[min_nonzero_index]  #这个值就是用来填充的值

        # 返回这个数在原矩阵中的索引
        min_d_index = nonzero_indices[1][min_nonzero_index]

        if min_d_index < R_S:
            min_p = s_R_d_index[min_d_index]  #最小差距是行
            flag_column = True  #此时要寻找最大差距的列
        else:
            min_p = s_C_d_index[min_d_index % R_S]  #最小车距是列，此时要寻找最大差距的行

        if flag_column:  # 如果要寻找最小差距的列
            # min_to_stuff_index = np.argmin(column_differ)
            min_to_stuff_index = np.nonzero(column_differ)[0][np.argmin(column_differ[np.nonzero(column_differ)])]
            if column_differ[min_to_stuff_index] > min_nonzero_d:  #如果需求大于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[min_p][min_to_stuff_index] += min_nonzero_d
                else:  #如果不是最后一次，则依次填充
                    if min_p != min_to_stuff_index:
                        copied_matrix[min_p][min_to_stuff_index] += min_nonzero_d  #列的差值大于待填充值， 直接填充待填充值
                    else:
                        column_differ_copy = np.copy(column_differ)
                        while min_nonzero_d > 0 and np.count_nonzero(column_differ_copy) != 0:  #再循环过程中肯定不会再取到对角线了
                            if column_differ_copy[min_p] != 0 and np.count_nonzero(column_differ_copy) == 1:
                                column_differ_copy[min_p] = column_differ[min_p]
                                if column_differ[min_p] >= min_nonzero_d:
                                    copied_matrix[min_p][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ_copy[min_p] -= min_nonzero_d
                                    min_nonzero_d = 0
                                else:
                                    copied_matrix[min_p][min_p] += column_differ[min_p]
                                    min_nonzero_d -= column_differ[min_p]
                                    column_differ_copy[min_p] = 0  # 该列剩余的差值
                                break
                            column_differ_copy[min_to_stuff_index] = np.iinfo(np.int32).max
                            second_largest_index = np.nonzero(column_differ_copy)[0][
                                np.argmin(column_differ_copy[np.nonzero(column_differ_copy)])]
                            # second_largest_index = np.argmin(column_differ)# 如果是对角线，则取第二大的值
                            if column_differ[second_largest_index] > min_nonzero_d:  #如果第二大的值大于待填充值，则填充待填充值
                                copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                column_differ_copy[second_largest_index] -= min_nonzero_d
                                min_nonzero_d = 0
                            else:  #待填充值大于列那个值
                                copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
                                min_nonzero_d -= column_differ[second_largest_index]
                                column_differ_copy[second_largest_index] = 0  #该列剩余的差值

            else:  #如果列需求小于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[min_p][min_to_stuff_index] += column_differ[min_to_stuff_index]
                else:  #如果不是最后一次，则依次填充
                    if min_p != min_to_stuff_index:  #先判断如果不是对角线，就一直填充，直到填充值为0或带
                        copied_matrix[min_p][min_to_stuff_index] += column_differ[min_to_stuff_index]
                        min_nonzero_d -= column_differ[min_to_stuff_index]  # 新的待填充值
                        column_differ[min_to_stuff_index] = 0  # 该列剩余的差值
                        while min_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:
                            # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                            # second_largest_index = np.argmin(column_differ)
                            second_largest_index = np.nonzero(column_differ)[0][
                                np.argmin(column_differ[np.nonzero(column_differ)])]
                            if second_largest_index != min_p:
                                if column_differ[second_largest_index] > min_nonzero_d:
                                    copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ[second_largest_index] -= min_nonzero_d
                                    min_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
                                    min_nonzero_d -= column_differ[second_largest_index]
                                    column_differ[second_largest_index] = 0  # 该列剩余的差值
                            else:

                                if np.count_nonzero(column_differ) == 1:
                                    pass
                                else:
                                    column_differ[second_largest_index] = np.iinfo(np.int32).max


                    else:  #如果是对角线，就换第二大的位置
                        column_differ_copy = np.copy(column_differ)

                        while min_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:
                            if column_differ_copy[min_p] != 0 and np.count_nonzero(column_differ_copy) == 1:
                                column_differ_copy[min_p] = column_differ[min_p]
                                if column_differ[min_p] >= min_nonzero_d:
                                    copied_matrix[min_p][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ_copy[min_p] -= min_nonzero_d
                                    min_nonzero_d = 0
                                else:
                                    copied_matrix[min_p][min_p] += column_differ[min_p]
                                    min_nonzero_d -= column_differ[min_p]
                                    column_differ_copy[min_p] = 0  # 该列剩余的差值
                                break
                            column_differ_copy[min_to_stuff_index] = np.iinfo(np.int32).max
                            # second_largest_index = np.argmin(column_differ)
                            second_largest_index = np.nonzero(column_differ_copy)[0][
                                np.argmin(column_differ_copy[np.nonzero(column_differ_copy)])]
                            if column_differ[second_largest_index] > min_nonzero_d:
                                copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                column_differ_copy[second_largest_index] -= min_nonzero_d
                                min_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
                                min_nonzero_d -= column_differ[second_largest_index]
                                column_differ_copy[second_largest_index] = 0  # 该列剩余的差值
                # 根据索引信息找到在原数组中的索引
        else:  #flag_column == False
            # max_to_stuff_index = np.argmin(row_differ)  # 获取行最大值的索引
            max_to_stuff_index = np.nonzero(row_differ)[0][np.argmin(row_differ[np.nonzero(row_differ)])]
            if row_differ[max_to_stuff_index] > min_nonzero_d:  # 如果需求大于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
                    copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
                else:  # 如果不是最后一次，则依次填充
                    if min_p != max_to_stuff_index:
                        copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d  # 列的差值大于待填充值， 直接填充待填充值
                    else:  #如果是对角线
                        row_differ_copy = np.copy(row_differ)

                        while min_nonzero_d > 0 and np.count_nonzero(row_differ_copy) != 0:
                            if row_differ_copy[min_p] != 0 and np.count_nonzero(row_differ_copy) == 1:
                                row_differ_copy[min_p] = row_differ[min_p]
                                if row_differ[min_p] >= min_nonzero_d:
                                    copied_matrix[min_p][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ_copy[min_p] -= min_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[min_p][min_p] += row_differ[min_p]
                                    min_nonzero_d -= row_differ[min_p]
                                    row_differ_copy[min_p] = 0  # 该列剩余的差值
                                break
                            row_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).max
                            second_largest_index = np.nonzero(row_differ_copy)[0][
                                np.argmin(row_differ_copy[np.nonzero(row_differ_copy)])]
                            # second_largest_index = np.argmin(row_differ)  # 如果是对角线，则取第二大的值
                            if row_differ[second_largest_index] > min_nonzero_d:  # 如果第二大的值大于待填充值，则填充待填充值
                                copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                row_differ_copy[second_largest_index] -= min_nonzero_d
                                min_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
                                min_nonzero_d -= row_differ[second_largest_index]
                                row_differ_copy[second_largest_index] = 0  # 该列剩余的差值

            else:  #如果需求小于等于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[max_to_stuff_index][min_p] += row_differ[max_to_stuff_index]
                else:  #如果不是最后一次，则依次填充
                    if min_p != max_to_stuff_index:  #先判断如果不是对角线，不是就一直填充，直到填充值为0或带
                        copied_matrix[max_to_stuff_index][min_p] += row_differ[max_to_stuff_index]
                        min_nonzero_d -= row_differ[max_to_stuff_index]  # 新的待填充值
                        row_differ[max_to_stuff_index] = 0  # 该列剩余的差值
                        row_d_copy = np.copy(row_differ)  #最开始的差值
                        while min_d_index > 0 and np.count_nonzero(row_differ) != 0:
                            # row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                            if np.count_nonzero(row_differ) == 1:
                                second_largest_index = np.nonzero(row_differ)[0][0]
                                if row_d_copy[second_largest_index] > min_nonzero_d:
                                    copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ[second_largest_index] -= min_nonzero_d
                                    min_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[second_largest_index][min_p] += row_d_copy[second_largest_index]
                                    min_nonzero_d -= row_d_copy[second_largest_index]
                                    row_differ[second_largest_index] = 0  # 该列剩余的差
                            else:
                                # second_largest_index = np.argmin(row_differ)
                                second_largest_index = np.nonzero(row_differ)[0][
                                    np.argmin(row_differ[np.nonzero(row_differ)])]
                            if second_largest_index != min_p:
                                if row_differ[second_largest_index] > min_nonzero_d:
                                    copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ[second_largest_index] -= min_nonzero_d
                                    max_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
                                    min_nonzero_d -= row_differ[second_largest_index]
                                    row_differ[second_largest_index] = 0  # 该列剩余的差
                            else:
                                if np.count_nonzero(row_differ) == 1:
                                    pass
                                else:
                                    row_differ[second_largest_index] = np.iinfo(np.int32).max

                            # max_to_stuff_index = second_largest_index
                            # row_differ[second_largest_index] = np.iinfo(np.int32).min


                    else:  #如果是对角线，就换第二大的位置
                        row_differ_copy = np.copy(row_differ)

                        while min_nonzero_d > 0 and np.count_nonzero(row_differ_copy) != 0:
                            if row_differ_copy[min_p] != 0 and np.count_nonzero(row_differ_copy) == 1:
                                row_differ_copy[min_p] = row_differ[min_p]
                                if row_differ[min_p] >= min_nonzero_d:
                                    copied_matrix[min_p][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ_copy[min_p] -= min_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[min_p][min_p] += row_differ[min_p]
                                    min_nonzero_d -= row_differ[min_p]
                                    row_differ_copy[min_p] = 0  # 该列剩余的差值
                                break

                            row_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).max
                            second_largest_index = np.argmin(row_differ_copy)

                            if row_differ[second_largest_index] > min_nonzero_d:
                                copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                row_differ_copy[second_largest_index] -= min_nonzero_d
                                min_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
                                min_nonzero_d -= row_differ[second_largest_index]
                                row_differ_copy[second_largest_index] = 0  # 该列剩余的差值

    return copied_matrix


def stuffing_max(matrix):
    copied_matrix = np.copy(matrix)
    while True:
        row_sum = copied_matrix.sum(axis=1)
        column_sum = copied_matrix.sum(axis=0)
        if np.array_equal(row_sum, column_sum):
            if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
                break
        max_value = max(np.maximum(row_sum, column_sum))
        flag_column = False  #寻找列为假
        row_differ = np.array([max_value] * R_S) - row_sum  #获取行的差
        column_differ = np.array([max_value] * R_S) - column_sum  #获取列的差

        s_R_d_index = np.argsort(row_differ)  #行差排序后的索引
        sort_row_differ = row_differ[s_R_d_index]  #行差排序后的结果

        s_C_d_index = np.argsort(column_differ)  #列差排序后的索引
        sort_column_differ = column_differ[s_C_d_index]  #列差排序后的结果

        # 合并排序后的行差和列差为一个新的数组
        combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)

        # 找到新数组非零元素的索引
        nonzero_indices = np.nonzero(combined_array)

        # 获取新数组非零元素
        nonzero_elements = combined_array[nonzero_indices]

        # 找到新数组非零最小值的索引和非零元素的最小值
        max_nonzero_index = np.argmax(nonzero_elements)
        max_nonzero_d = nonzero_elements[max_nonzero_index]  #这个值就是用来填充的值

        # 返回这个数在原矩阵中的索引
        max_d_index = nonzero_indices[1][max_nonzero_index]

        if max_d_index < R_S:
            max_p = s_R_d_index[max_d_index]  #最小差距是行
            flag_column = True  #此时要寻找最大差距的列
        else:
            max_p = s_C_d_index[max_d_index % R_S]  #最小车距是列，此时要寻找最大差距的行

        if flag_column:  # 如果要寻找最大差距的列
            max_to_stuff_index = np.argmax(column_differ)
            if column_differ[max_to_stuff_index] > max_nonzero_d:  #如果需求大于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[max_p][max_to_stuff_index] += max_nonzero_d
                else:  #如果不是最后一次，则依次填充
                    if max_p != max_to_stuff_index:
                        copied_matrix[max_p][max_to_stuff_index] += max_nonzero_d  #列的差值大于待填充值， 直接填充待填充值
                        # column_differ[max_to_stuff_index] -= max_nonzero_d# 该列剩余的差值
                        # max_nonzero_d = column_differ[max_to_stuff_index]# 新的待填充值
                        # row_differ[max_p] = 0  # 该行的差值被填平
                        # while True:
                        #     row_differ_copy = np.copy(row_differ)
                        #     while True:
                        #         second_largest_index = np.argmax(row_differ_copy)
                        #         if max_p != second_largest_index or np.count_nonzero(row_differ) == 1:
                        #             break
                        #     copied_matrix[max_p][second_largest_index] += max_nonzero_d
                        #     if column_differ[max_to_stuff_index]==0 or np.count_nonzero(row_differ) == 0:
                        #         break
                    else:
                        ## 先获取索引
                        # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                        # second_largest_index = np.argmax(column_differ)
                        ## 填充
                        column_differ_copy = np.copy(column_differ)
                        while max_nonzero_d > 0 and np.count_nonzero(column_differ_copy) != 0:  #再循环过程中肯定不会再取到对角线了
                            if column_differ_copy[max_p] != 0 and np.count_nonzero(column_differ_copy) == 1:
                                column_differ_copy[max_p] = column_differ[max_p]
                                if column_differ[max_p] >= max_nonzero_d:
                                    copied_matrix[max_p][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ_copy[max_p] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[max_p][max_p] += column_differ[max_p]
                                    max_nonzero_d -= column_differ[max_p]
                                    column_differ_copy[max_p] = 0  # 该列剩余的差值
                                break
                            column_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).min
                            second_largest_index = np.argmax(column_differ_copy)  # 如果是对角线，则取第二大的值
                            if column_differ[second_largest_index] > max_nonzero_d:  #如果第二大的值大于待填充值，则填充待填充值
                                copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                column_differ_copy[second_largest_index] -= max_nonzero_d
                                max_nonzero_d = 0
                            else:  #待填充值大于列那个值
                                copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
                                max_nonzero_d -= column_differ[second_largest_index]
                                column_differ_copy[second_largest_index] = 0  #该列剩余的差值

            else:  #如果列需求小于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[max_p][max_to_stuff_index] += column_differ[max_to_stuff_index]
                else:  #如果不是最后一次，则依次填充
                    if max_p != max_to_stuff_index:  #先判断如果不是对角线，就一直填充，直到填充值为0或带
                        copied_matrix[max_p][max_to_stuff_index] += column_differ[max_to_stuff_index]
                        max_nonzero_d -= column_differ[max_to_stuff_index]  # 新的待填充值
                        column_differ[max_to_stuff_index] = 0  # 该列剩余的差值
                        while max_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:
                            # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                            second_largest_index = np.argmax(column_differ)
                            if second_largest_index != max_p:
                                if column_differ[second_largest_index] > max_nonzero_d:
                                    copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ[second_largest_index] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
                                    max_nonzero_d -= column_differ[second_largest_index]
                                    column_differ[second_largest_index] = 0  # 该列剩余的差值
                            else:
                                # max_to_stuff_index = second_largest_index
                                # arr_copy_column = np.copy(column_differ)
                                # arr_copy_column[second_largest_index] = -np.inf
                                # second_largest_index = np.argmax(arr_copy_column)
                                if np.count_nonzero(column_differ) == 1:
                                    pass
                                else:
                                    column_differ[second_largest_index] = np.iinfo(np.int32).min


                    else:  #如果是对角线，就换第二大的位置
                        column_differ_copy = np.copy(column_differ)

                        while max_nonzero_d > 0 and np.count_nonzero(column_differ_copy) != 0:
                            if column_differ_copy[max_p] != 0 and np.count_nonzero(column_differ_copy) == 1:
                                column_differ_copy[max_p] = column_differ[max_p]
                                if column_differ[max_p] >= max_nonzero_d:
                                    copied_matrix[max_p][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    column_differ_copy[max_p] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[max_p][max_p] += column_differ[max_p]
                                    max_nonzero_d -= column_differ[max_p]
                                    column_differ_copy[max_p] = 0  # 该列剩余的差值
                                break
                            column_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).min
                            second_largest_index = np.argmax(column_differ_copy)
                            if column_differ[second_largest_index] > max_nonzero_d:
                                copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                column_differ_copy[second_largest_index] -= max_nonzero_d
                                max_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
                                max_nonzero_d -= column_differ[second_largest_index]
                                column_differ_copy[second_largest_index] = 0  # 该列剩余的差值
                # 根据索引信息找到在原数组中的索引
        else:  #flag_column == False
            max_to_stuff_index = np.argmax(row_differ)  # 获取行最大值的索引
            if row_differ[max_to_stuff_index] > max_nonzero_d:  # 如果需求大于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
                    copied_matrix[max_to_stuff_index][max_p] += max_nonzero_d
                else:  # 如果不是最后一次，则依次填充
                    if max_p != max_to_stuff_index:
                        copied_matrix[max_to_stuff_index][max_p] += max_nonzero_d  # 列的差值大于待填充值， 直接填充待填充值
                    else:  #如果是对角线
                        row_differ_copy = np.copy(row_differ)

                        while max_nonzero_d > 0 and np.count_nonzero(row_differ_copy) != 0:
                            if row_differ_copy[max_p] != 0 and np.count_nonzero(row_differ_copy) == 1:
                                row_differ_copy[max_p] = row_differ[max_p]
                                if row_differ[max_p] >= max_nonzero_d:
                                    copied_matrix[max_p][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ_copy[max_p] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[max_p][max_p] += row_differ[max_p]
                                    max_nonzero_d -= row_differ[max_p]
                                    row_differ_copy[max_p] = 0  # 该列剩余的差值
                                break
                            row_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).min
                            second_largest_index = np.argmax(row_differ_copy)  # 如果是对角线，则取第二大的值
                            if row_differ[second_largest_index] > max_nonzero_d:  # 如果第二大的值大于待填充值，则填充待填充值
                                copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                row_differ_copy[second_largest_index] -= max_nonzero_d
                                max_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
                                max_nonzero_d -= row_differ[second_largest_index]
                                row_differ_copy[second_largest_index] = 0  # 该列剩余的差值

            else:  #如果需求小于等于被填充值
                if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  #如果是最后一次循环，则直接填充
                    copied_matrix[max_to_stuff_index][max_p] += row_differ[max_to_stuff_index]
                else:  #如果不是最后一次，则依次填充
                    if max_p != max_to_stuff_index:  #先判断如果不是对角线，不是就一直填充，直到填充值为0或带
                        copied_matrix[max_to_stuff_index][max_p] += row_differ[max_to_stuff_index]
                        max_nonzero_d -= row_differ[max_to_stuff_index]  # 新的待填充值
                        row_differ[max_to_stuff_index] = 0  # 该列剩余的差值
                        row_d_copy = np.copy(row_differ)  #最开始的差值
                        while max_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
                            # row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                            if np.count_nonzero(row_differ) == 1:
                                second_largest_index = np.nonzero(row_differ)[0][0]
                                if row_d_copy[second_largest_index] > max_nonzero_d:
                                    copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ[second_largest_index] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[second_largest_index][max_p] += row_d_copy[second_largest_index]
                                    max_nonzero_d -= row_d_copy[second_largest_index]
                                    row_differ[second_largest_index] = 0  # 该列剩余的差
                            else:
                                second_largest_index = np.argmax(row_differ)

                            if second_largest_index != max_p:
                                if row_differ[second_largest_index] > max_nonzero_d:
                                    copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ[second_largest_index] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:  # 待填充值大于列那个值
                                    copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
                                    max_nonzero_d -= row_differ[second_largest_index]
                                    row_differ[second_largest_index] = 0  # 该列剩余的差
                            else:
                                if np.count_nonzero(row_differ) == 1:
                                    pass
                                else:
                                    row_differ[second_largest_index] = np.iinfo(np.int32).min

                            # max_to_stuff_index = second_largest_index
                            # row_differ[second_largest_index] = np.iinfo(np.int32).min


                    else:  #如果是对角线，就换第二大的位置
                        row_differ_copy = np.copy(row_differ)

                        while max_nonzero_d > 0 and np.count_nonzero(row_differ_copy) != 0:

                            if row_differ_copy[max_p] != 0 and np.count_nonzero(row_differ_copy) == 1:
                                row_differ_copy[max_p] = row_differ[max_p]

                                if row_differ[max_p] >= max_nonzero_d:
                                    copied_matrix[max_p][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                    row_differ_copy[max_p] -= max_nonzero_d
                                    max_nonzero_d = 0
                                else:
                                    copied_matrix[max_p][max_p] += row_differ[max_p]
                                    max_nonzero_d -= row_differ[max_p]
                                    row_differ_copy[max_p] = 0  # 该列剩余的差值
                                break
                            row_differ_copy[max_to_stuff_index] = np.iinfo(np.int32).min
                            second_largest_index = np.argmax(row_differ_copy)
                            if row_differ[second_largest_index] > max_nonzero_d:
                                copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
                                row_differ_copy[second_largest_index] -= max_nonzero_d
                                max_nonzero_d = 0
                            else:  # 待填充值大于列那个值
                                copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
                                max_nonzero_d -= row_differ[second_largest_index]
                                row_differ_copy[second_largest_index] = 0  # 该列剩余的差值

    return copied_matrix


# def stuffing_new_min(matrix):
#     copied_matrix = np.copy(matrix)
#     while True:
#         row_sum = copied_matrix.sum(axis=1)
#         column_sum = copied_matrix.sum(axis=0)
#         if np.array_equal(row_sum, column_sum):
#             if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
#                 break
#         max_value = max(np.maximum(row_sum, column_sum))
#         flag_column = False #寻找列为假
#         row_differ = np.array([max_value] * R_S) - row_sum #获取行的差
#         column_differ = np.array([max_value] * R_S) - column_sum #获取列的差
#
#         s_R_d_index = np.argsort(row_differ) #行差排序后的索引
#         sort_row_differ = row_differ[s_R_d_index] #行差排序后的结果
#
#         s_C_d_index = np.argsort(column_differ) #列差排序后的索引
#         sort_column_differ = column_differ[s_C_d_index] #列差排序后的结果
#
#         # 合并排序后的行差和列差为一个新的数组
#         combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)
#
#         # 找到新数组非零元素的索引
#         nonzero_indices = np.nonzero(combined_array)
#
#         # 获取新数组非零元素
#         nonzero_elements = combined_array[nonzero_indices]
#
#         # 找到新数组非零最小值的索引和非零元素的最小值
#         min_nonzero_index = np.argmin(nonzero_elements)
#         min_nonzero_d = nonzero_elements[min_nonzero_index] #这个值就是用来填充的值
#
#         # 返回这个数在原矩阵中的索引
#         min_d_index = nonzero_indices[1][min_nonzero_index]
#
#         if min_d_index < R_S:
#             min_p = s_R_d_index[min_d_index] #最小差距是行
#             flag_column = True #此时要寻找最大差距的列
#         else:
#             min_p = s_C_d_index[min_d_index % R_S] #最小车距是列，此时要寻找最大差距的行
#
#         if flag_column:  # 如果要寻找最小差距的列
#             # min_to_stuff_index = np.argmin(column_differ)
#             min_to_stuff_index=np.nonzero(column_differ)[0][np.argmin(column_differ[np.nonzero(column_differ)])]
#             if column_differ[min_to_stuff_index] > min_nonzero_d: #如果需求大于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1: #如果是最后一次循环，则直接填充
#                     copied_matrix[min_p][min_to_stuff_index] += min_nonzero_d
#                 else: #如果不是最后一次，则依次填充
#                     if min_p != min_to_stuff_index:
#                         copied_matrix[min_p][min_to_stuff_index] += min_nonzero_d #列的差值大于待填充值， 直接填充待填充值
#                     else:
#                         while min_nonzero_d>0 and np.count_nonzero(column_differ)!=0: #再循环过程中肯定不会再取到对角线了
#                             column_differ[min_to_stuff_index] = np.iinfo(np.int32).max
#                             second_largest_index = np.nonzero(column_differ)[0][np.argmin(column_differ[np.nonzero(column_differ)])]
#                             # second_largest_index = np.argmin(column_differ)# 如果是对角线，则取第二大的值
#                             if column_differ[second_largest_index] > min_nonzero_d:#如果第二大的值大于待填充值，则填充待填充值
#                                 copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 column_differ[second_largest_index] -= min_nonzero_d
#                                 min_nonzero_d = 0
#                             else:#待填充值大于列那个值
#                                 copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
#                                 min_nonzero_d -= column_differ[second_largest_index]
#                                 column_differ[second_largest_index] = 0  #该列剩余的差值
#
#             else:#如果列需求小于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1: #如果是最后一次循环，则直接填充
#                     copied_matrix[min_p][min_to_stuff_index] += column_differ[min_to_stuff_index]
#                 else: #如果不是最后一次，则依次填充
#                     if min_p != min_to_stuff_index: #先判断如果不是对角线，就一直填充，直到填充值为0或带
#                         copied_matrix[min_p][min_to_stuff_index] += column_differ[min_to_stuff_index]
#                         min_nonzero_d -= column_differ[min_to_stuff_index]# 新的待填充值
#                         column_differ[min_to_stuff_index] = 0# 该列剩余的差值
#                         while min_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:
#                             # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             # second_largest_index = np.argmin(column_differ)
#                             second_largest_index = np.nonzero(column_differ)[0][np.argmin(column_differ[np.nonzero(column_differ)])]
#                             if second_largest_index != min_p:
#                                 if column_differ[second_largest_index] > min_nonzero_d:
#                                     copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     column_differ[second_largest_index] -= min_nonzero_d
#                                     min_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
#                                     min_nonzero_d -= column_differ[second_largest_index]
#                                     column_differ[second_largest_index] = 0  # 该列剩余的差值
#                             else:
#
#                                 if np.count_nonzero(column_differ) == 1:
#                                     pass
#                                 else:
#                                     column_differ[second_largest_index] = np.iinfo(np.int32).max
#
#
#                     else: #如果是对角线，就换第二大的位置
#                         while min_d_index > 0 or np.count_nonzero(column_differ) != 0:
#                             column_differ[min_to_stuff_index] = np.iinfo(np.int32).max
#                             # second_largest_index = np.argmin(column_differ)
#                             second_largest_index = np.nonzero(column_differ)[0][np.argmin(column_differ[np.nonzero(column_differ)])]
#                             if column_differ[second_largest_index] > min_nonzero_d:
#                                 copied_matrix[min_p][second_largest_index] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 column_differ[second_largest_index] -= min_nonzero_d
#                                 min_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[min_p][second_largest_index] += column_differ[second_largest_index]
#                                 min_nonzero_d -= column_differ[second_largest_index]
#                                 column_differ[second_largest_index] = 0  # 该列剩余的差值
#                 # 根据索引信息找到在原数组中的索引
#         else: #flag_column == False
#             # max_to_stuff_index = np.argmin(row_differ)  # 获取行最大值的索引
#             max_to_stuff_index=np.nonzero(row_differ)[0][np.argmin(row_differ[np.nonzero(row_differ)])]
#             if row_differ[max_to_stuff_index] > min_nonzero_d:  # 如果需求大于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
#                     copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
#                 else:  # 如果不是最后一次，则依次填充
#                     if min_p != max_to_stuff_index:
#                         copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d  # 列的差值大于待填充值， 直接填充待填充值
#                     else:#如果是对角线
#                         while min_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
#                             row_differ[max_to_stuff_index] = np.iinfo(np.int32).max
#                             second_largest_index = np.nonzero(row_differ)[0][np.argmin(row_differ[np.nonzero(row_differ)])]
#                             # second_largest_index = np.argmin(row_differ)  # 如果是对角线，则取第二大的值
#                             if row_differ[second_largest_index] > min_nonzero_d:  # 如果第二大的值大于待填充值，则填充待填充值
#                                 copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 row_differ[second_largest_index] -= min_nonzero_d
#                                 min_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
#                                 min_nonzero_d -= row_differ[second_largest_index]
#                                 row_differ[second_largest_index] = 0  # 该列剩余的差值
#
#             else:#如果需求小于等于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1: #如果是最后一次循环，则直接填充
#                     copied_matrix[max_to_stuff_index][min_p] += row_differ[max_to_stuff_index]
#                 else: #如果不是最后一次，则依次填充
#                     if min_p != max_to_stuff_index: #先判断如果不是对角线，不是就一直填充，直到填充值为0或带
#                         copied_matrix[max_to_stuff_index][min_p] += row_differ[max_to_stuff_index]
#                         min_nonzero_d -= row_differ[max_to_stuff_index]# 新的待填充值
#                         row_differ[max_to_stuff_index] = 0# 该列剩余的差值
#                         row_d_copy = np.copy(row_differ) #最开始的差值
#                         while min_d_index > 0 and np.count_nonzero(row_differ) != 0:
#                             # row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             if np.count_nonzero(row_differ) == 1:
#                                 second_largest_index = np.nonzero(row_differ)[0][0]
#                                 if row_d_copy[second_largest_index] > min_nonzero_d:
#                                     copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     row_differ[second_largest_index] -= min_nonzero_d
#                                     min_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[second_largest_index][min_p] += row_d_copy[second_largest_index]
#                                     min_nonzero_d -= row_d_copy[second_largest_index]
#                                     row_differ[second_largest_index] = 0  # 该列剩余的差
#                             else:
#                                 # second_largest_index = np.argmin(row_differ)
#                                 second_largest_index = np.nonzero(row_differ)[0][np.argmin(row_differ[np.nonzero(row_differ)])]
#                             if second_largest_index != min_p:
#                                 if row_differ[second_largest_index] > min_nonzero_d:
#                                     copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     row_differ[second_largest_index] -= min_nonzero_d
#                                     max_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
#                                     min_nonzero_d -= row_differ[second_largest_index]
#                                     row_differ[second_largest_index] = 0  # 该列剩余的差
#                             else:
#                                 if np.count_nonzero(row_differ) == 1:
#                                     pass
#                                 else:
#                                     row_differ[second_largest_index] = np.iinfo(np.int32).max
#
#                             # max_to_stuff_index = second_largest_index
#                             # row_differ[second_largest_index] = np.iinfo(np.int32).min
#
#
#                     else: #如果是对角线，就换第二大的位置
#                         while min_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
#                             row_differ[max_to_stuff_index] = np.iinfo(np.int32).max
#                             second_largest_index = np.argmin(row_differ)
#
#                             if row_differ[second_largest_index] > min_nonzero_d:
#                                 copied_matrix[second_largest_index][min_p] += min_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 row_differ[second_largest_index] -= min_nonzero_d
#                                 min_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[second_largest_index][min_p] += row_differ[second_largest_index]
#                                 min_nonzero_d -= row_differ[second_largest_index]
#                                 row_differ[second_largest_index] = 0  # 该列剩余的差值
#
#
#     return copied_matrix
#

# def stuffing_max(matrix):
#     copied_matrix = np.copy(matrix)
#     while True:
#         row_sum = copied_matrix.sum(axis=1)
#         column_sum = copied_matrix.sum(axis=0)
#         if np.array_equal(row_sum, column_sum):
#             if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
#                 break
#         max_value = max(np.maximum(row_sum, column_sum))
#         flag_column = False  # 寻找列为假
#         row_differ = np.array([max_value] * R_S) - row_sum  # 获取行的差
#         column_differ = np.array([max_value] * R_S) - column_sum  # 获取列的差
#
#         s_R_d_index = np.argsort(row_differ)  # 行差排序后的索引
#         sort_row_differ = row_differ[s_R_d_index]  # 行差排序后的结果
#
#         s_C_d_index = np.argsort(column_differ)  # 列差排序后的索引
#         sort_column_differ = column_differ[s_C_d_index]  # 列差排序后的结果
#
#         # 合并排序后的行差和列差为一个新的数组
#         combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)
#
#         # 找到新数组非零元素的索引
#         nonzero_indices = np.nonzero(combined_array)
#
#         # 获取新数组非零元素
#         nonzero_elements = combined_array[nonzero_indices]
#
#         # 找到新数组非零最小值的索引和非零元素的最小值
#         max_nonzero_index = np.argmax(nonzero_elements)
#         max_nonzero_d = nonzero_elements[max_nonzero_index]  # 这个值就是用来填充的值
#
#         # 返回这个数在原矩阵中的索引
#         max_d_index = nonzero_indices[1][max_nonzero_index]
#
#         if max_d_index < R_S:
#             max_p = s_R_d_index[max_d_index]  # 最小差距是行
#             flag_column = True  # 此时要寻找最大差距的列
#         else:
#             max_p = s_C_d_index[max_d_index % R_S]  # 最小车距是列，此时要寻找最大差距的行
#
#         if flag_column:  # 如果要寻找最大差距的列
#             max_to_stuff_index = np.argmax(column_differ)
#             if column_differ[max_to_stuff_index] > max_nonzero_d:  # 如果需求大于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
#                     copied_matrix[max_p][max_to_stuff_index] += max_nonzero_d
#                 else:  # 如果不是最后一次，则依次填充
#                     if max_p != max_to_stuff_index:
#                         copied_matrix[max_p][max_to_stuff_index] += max_nonzero_d  # 列的差值大于待填充值， 直接填充待填充值
#                         # column_differ[max_to_stuff_index] -= max_nonzero_d# 该列剩余的差值
#                         # max_nonzero_d = column_differ[max_to_stuff_index]# 新的待填充值
#                         # row_differ[max_p] = 0  # 该行的差值被填平
#                         # while True:
#                         #     row_differ_copy = np.copy(row_differ)
#                         #     while True:
#                         #         second_largest_index = np.argmax(row_differ_copy)
#                         #         if max_p != second_largest_index or np.count_nonzero(row_differ) == 1:
#                         #             break
#                         #     copied_matrix[max_p][second_largest_index] += max_nonzero_d
#                         #     if column_differ[max_to_stuff_index]==0 or np.count_nonzero(row_differ) == 0:
#                         #         break
#                     else:
#                         ## 先获取索引
#                         # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                         # second_largest_index = np.argmax(column_differ)
#                         ## 填充
#                         while max_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:  # 再循环过程中肯定不会再取到对角线了
#                             column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             second_largest_index = np.argmax(column_differ)  # 如果是对角线，则取第二大的值
#                             if column_differ[second_largest_index] > max_nonzero_d:  # 如果第二大的值大于待填充值，则填充待填充值
#                                 copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 column_differ[second_largest_index] -= max_nonzero_d
#                                 max_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
#                                 max_nonzero_d -= column_differ[second_largest_index]
#                                 column_differ[second_largest_index] = 0  # 该列剩余的差值
#
#             else:  # 如果列需求小于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
#                     copied_matrix[max_p][max_to_stuff_index] += column_differ[max_to_stuff_index]
#                 else:  # 如果不是最后一次，则依次填充
#                     if max_p != max_to_stuff_index:  # 先判断如果不是对角线，就一直填充，直到填充值为0或带
#                         copied_matrix[max_p][max_to_stuff_index] += column_differ[max_to_stuff_index]
#                         max_nonzero_d -= column_differ[max_to_stuff_index]  # 新的待填充值
#                         column_differ[max_to_stuff_index] = 0  # 该列剩余的差值
#                         while max_nonzero_d > 0 and np.count_nonzero(column_differ) != 0:
#                             # column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             second_largest_index = np.argmax(column_differ)
#                             if second_largest_index != max_p:
#                                 if column_differ[second_largest_index] > max_nonzero_d:
#                                     copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     column_differ[second_largest_index] -= max_nonzero_d
#                                     max_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
#                                     max_nonzero_d -= column_differ[second_largest_index]
#                                     column_differ[second_largest_index] = 0  # 该列剩余的差值
#                             else:
#                                 # max_to_stuff_index = second_largest_index
#                                 # arr_copy_column = np.copy(column_differ)
#                                 # arr_copy_column[second_largest_index] = -np.inf
#                                 # second_largest_index = np.argmax(arr_copy_column)
#                                 if np.count_nonzero(column_differ) == 1:
#                                     pass
#                                 else:
#                                     column_differ[second_largest_index] = np.iinfo(np.int32).min
#
#
#                     else:  # 如果是对角线，就换第二大的位置
#                         while max_nonzero_d > 0 or np.count_nonzero(column_differ) != 0:
#                             column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             second_largest_index = np.argmax(column_differ)
#                             if column_differ[second_largest_index] > max_nonzero_d:
#                                 copied_matrix[max_p][second_largest_index] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 column_differ[second_largest_index] -= max_nonzero_d
#                                 max_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[max_p][second_largest_index] += column_differ[second_largest_index]
#                                 max_nonzero_d -= column_differ[second_largest_index]
#                                 column_differ[second_largest_index] = 0  # 该列剩余的差值
#                 # 根据索引信息找到在原数组中的索引
#         else:  # flag_column == False
#             max_to_stuff_index = np.argmax(row_differ)  # 获取行最大值的索引
#             if row_differ[max_to_stuff_index] > max_nonzero_d:  # 如果需求大于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
#                     copied_matrix[max_to_stuff_index][max_p] += max_nonzero_d
#                 else:  # 如果不是最后一次，则依次填充
#                     if max_p != max_to_stuff_index:
#                         copied_matrix[max_to_stuff_index][max_p] += max_nonzero_d  # 列的差值大于待填充值， 直接填充待填充值
#                     else:  # 如果是对角线
#                         while max_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
#                             row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             second_largest_index = np.argmax(row_differ)  # 如果是对角线，则取第二大的值
#                             if row_differ[second_largest_index] > max_nonzero_d:  # 如果第二大的值大于待填充值，则填充待填充值
#                                 copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 row_differ[second_largest_index] -= max_nonzero_d
#                                 max_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
#                                 max_nonzero_d -= row_differ[second_largest_index]
#                                 row_differ[second_largest_index] = 0  # 该列剩余的差值
#
#             else:  # 如果需求小于等于被填充值
#                 if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:  # 如果是最后一次循环，则直接填充
#                     copied_matrix[max_to_stuff_index][max_p] += row_differ[max_to_stuff_index]
#                 else:  # 如果不是最后一次，则依次填充
#                     if max_p != max_to_stuff_index:  # 先判断如果不是对角线，不是就一直填充，直到填充值为0或带
#                         copied_matrix[max_to_stuff_index][max_p] += row_differ[max_to_stuff_index]
#                         max_nonzero_d -= row_differ[max_to_stuff_index]  # 新的待填充值
#                         row_differ[max_to_stuff_index] = 0  # 该列剩余的差值
#                         row_d_copy = np.copy(row_differ)  # 最开始的差值
#                         while max_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
#                             # row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             if np.count_nonzero(row_differ) == 1:
#                                 second_largest_index = np.nonzero(row_differ)[0][0]
#                                 if row_d_copy[second_largest_index] > max_nonzero_d:
#                                     copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     row_differ[second_largest_index] -= max_nonzero_d
#                                     max_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[second_largest_index][max_p] += row_d_copy[second_largest_index]
#                                     max_nonzero_d -= row_d_copy[second_largest_index]
#                                     row_differ[second_largest_index] = 0  # 该列剩余的差
#                             else:
#                                 second_largest_index = np.argmax(row_differ)
#                             if second_largest_index != max_p:
#                                 if row_differ[second_largest_index] > max_nonzero_d:
#                                     copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                     row_differ[second_largest_index] -= max_nonzero_d
#                                     max_nonzero_d = 0
#                                 else:  # 待填充值大于列那个值
#                                     copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
#                                     max_nonzero_d -= row_differ[second_largest_index]
#                                     row_differ[second_largest_index] = 0  # 该列剩余的差
#                             else:
#                                 if np.count_nonzero(row_differ) == 1:
#                                     pass
#                                 else:
#                                     row_differ[second_largest_index] = np.iinfo(np.int32).min
#
#                             # max_to_stuff_index = second_largest_index
#                             # row_differ[second_largest_index] = np.iinfo(np.int32).min
#
#
#                     else:  # 如果是对角线，就换第二大的位置
#                         while max_nonzero_d > 0 and np.count_nonzero(row_differ) != 0:
#                             row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
#                             second_largest_index = np.argmax(row_differ)
#                             if row_differ[second_largest_index] > max_nonzero_d:
#                                 copied_matrix[second_largest_index][max_p] += max_nonzero_d  # 为列的差值大于待填充值， 直接填充待填充值
#                                 row_differ[second_largest_index] -= max_nonzero_d
#                                 max_nonzero_d = 0
#                             else:  # 待填充值大于列那个值
#                                 copied_matrix[second_largest_index][max_p] += row_differ[second_largest_index]
#                                 max_nonzero_d -= row_differ[second_largest_index]
#                                 row_differ[second_largest_index] = 0  # 该列剩余的差值
#
#     return copied_matrix


def Bvn_composition(stuffing):
    tensor_matrix = tf.convert_to_tensor(stuffing)
    expanded_tensor = tf.expand_dims(tensor_matrix, axis=0)
    t = tf.cast(expanded_tensor, dtype=tf.float32)
    permutation_matrix, coefficient = bvn.bvn(t, 1000)
    return permutation_matrix, coefficient


# 按比例分配
def OXC_connect_proportion(matrix, p, c):
    symmetric_matrix = make_symmetric(matrix)
    copied_matrix = np.copy(matrix)
    connection_matrix = np.zeros((R_S, R_S))
    sum_c = tf.reduce_sum(c)  # 系数之和
    inform_c = c / sum_c  # 系数比例
    rack_num = tf.round(inform_c * (Np / 2))  # 需要的矩阵数
    if rack_num[0][0] == 0:
        rack_num = tf.tensor_scatter_nd_update(rack_num, [[0, i] for i in range(Np // 2)], tf.ones((Np // 2,)))
        rack_num = rack_num[:, :Np // 2]
        p = p[:, :Np // 2, :, :]

    if tf.reduce_sum(rack_num) >= Np / 2:
        sum_ = tf.constant(0, dtype=tf.float32)
        for i in range(rack_num.shape[1]):
            sum_ += rack_num[0][i]
            if sum_ == (Np / 2):
                count = i + 1
                break
            if (Np / 2) - 1 < sum_ < (Np / 2):
                sliced_rack_num = rack_num[0, :i + 1]  #取出累加的这几个元素
                # 计算结果
                value_ = (Np / 2) - tf.reduce_sum(sliced_rack_num)  #求出差距
                rack_num = tf.tensor_scatter_nd_update(rack_num, tf.constant([[0, i + 1]]), [value_])  #赋值给下一个元素
                count = i + 2  #如果出现小数的情况则多取一个元素，因此+2
                break
        p_new = p[:, :count, :, :]  # 用于填充的的矩阵
        rack_new = rack_num[:, :count]  # 每个矩阵对应的个数

    else:  # 需要的矩阵数大于所有的rack_num之和
        for i in range(rack_num.shape[1]):
            differ = Np / 2 - tf.reduce_sum(rack_num)  # 求出差了多少
            if differ == 0:
                break
            if rack_num[0, i] < differ:
                rack_num_0i = rack_num[0][i] + rack_num[0][i]
                rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
            else:
                d_inter = int(differ)
                d_decimal = differ - int(differ)
                if isinstance(differ, int):
                    rack_num_0i = rack_num[0][i] + differ
                    rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
                else:
                    rack_num_0i = rack_num[0][i] + d_inter
                    rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
                    selected_matrix = p[:, i:i + 1, :, :]
                    # 将选中的矩阵添加在 p 的末尾
                    d_decimal = tf.constant(d_decimal)
                    d_decimal = tf.expand_dims(tf.expand_dims(d_decimal, axis=0), axis=0)
                    p = tf.concat([p, selected_matrix], axis=1)
                    rack_num = tf.concat([rack_num, d_decimal], axis=1)
        p_new = tf.Variable(tf.identity(p))
        rack_new = tf.Variable(tf.identity(rack_num))

    # 依次配置
    flag_int = True
    for _, n in enumerate(rack_new[0]):
        n_inter = int(n)
        n_decimal = n - int(n)
        count_num = 0
        while n_inter:
            p_matrix = p_new[0][_]
            oxc_indices = [tuple(sublist) for sublist in
                           tf.where(tf.not_equal(p_matrix, 0)).numpy()]  # 找出permutation矩阵中的非零元素并返回索引
            sorted_oxc_indices = [tuple(sorted(pair)) for pair in oxc_indices]
            for connect in sorted_oxc_indices:
                connection_matrix[connect] += 1
            count_num += 1  # 用完一个矩阵加一
            if count_num == n_inter:  # 当用的矩阵数量等于该矩阵的系数所对应的矩阵个数时跳出，取另一个矩阵，接着循环n次
                break
        if n_decimal:
            flag_int = False
            # 求出目前的连接关系
            trans_matrix = np.transpose(connection_matrix)
            connection_matrix = trans_matrix + connection_matrix
            #check
            # flag_connect_number = 1
            # for i in range(connection_matrix.shape[0]):
            #     sum_ = sum(connection_matrix[i])
            #     if sum_ != 48:
            #         flag_connect_number = 0
            # 判断是否有数据没有连接
            non_zero_indices = (symmetric_matrix > 0) & (connection_matrix == 0)  # 查看是否存在有数据但没有连接的情况
            remain_matrix = np.zeros_like(symmetric_matrix)  # 存在剩余需求
            remain_matrix[non_zero_indices] = symmetric_matrix[non_zero_indices]  # 剩余位置的需求
            remain_matrix = np.triu(remain_matrix)
            for i in range(int(R_S / 2)):
                if np.all(remain_matrix == -1):  # 如果没有剩余数据
                    if np.all(copied_matrix == -1):
                        break
                    else:
                        max_P = np.unravel_index(np.argmax(copied_matrix), copied_matrix.shape)  # 将扁平索引转为多维索引
                        row_index, col_index = max_P
                        connection_matrix[row_index][col_index] += 1
                        connection_matrix[col_index][row_index] += 1
                        # 将指定行和列的值赋为0
                        copied_matrix[:, col_index] = -1  # 将列的值赋为0
                        copied_matrix[row_index, :] = -1
                        copied_matrix[col_index, :] = -1  # 将列的值赋为0
                        copied_matrix[:, row_index] = -1
                else:  # 如果有剩余值
                    x, y = np.unravel_index(np.argmax(remain_matrix), remain_matrix.shape)
                    if x != y:
                        connection_matrix[x][y] += 1
                        connection_matrix[y][x] += 1
                        remain_matrix[:, y] = -1  # 将列的值赋为0
                        remain_matrix[x, :] = -1
                        remain_matrix[:, x] = -1  # 将列的值赋为0
                        remain_matrix[y, :] = -1
                        copied_matrix[:, y] = -1  # 将列的值赋为0
                        copied_matrix[x, :] = -1
                        copied_matrix[y, :] = -1  # 将列的值赋为0
                        copied_matrix[:, x] = -1
                    else:
                        remain_matrix[x][y] = -1
                        x, y = np.unravel_index(np.argmax(remain_matrix), remain_matrix.shape)
                        connection_matrix[x][y] += 1
                        connection_matrix[y][x] += 1
                        remain_matrix[:, y] = -1  # 将列的值赋为0
                        remain_matrix[x, :] = -1
                        remain_matrix[:, x] = -1  # 将列的值赋为0
                        remain_matrix[y, :] = -1
                        copied_matrix[:, y] = -1  # 将列的值赋为0
                        copied_matrix[x, :] = -1
                        copied_matrix[y, :] = -1  # 将列的值赋为0
                        copied_matrix[:, x] = -1

    if flag_int:
        trans_matrix = np.transpose(connection_matrix)
        connection_matrix = trans_matrix + connection_matrix

    return connection_matrix


# #全走一遍再按比例
def OXC_connect_all(matrix, p, c):
    symmetric_matrix = make_symmetric(matrix)
    copied_matrix = np.copy(matrix)
    p1 = tf.identity(p)
    connection_matrix = np.zeros((R_S, R_S))
    rack_num1 = tf.ones_like(c)
    sum_c = tf.reduce_sum(c)  #系数之和
    inform_c = c / sum_c  #系数比例
    rack_num = tf.round(inform_c * (Np / 2))  #需要的矩阵数
    if Np / 2 > tf.reduce_sum(rack_num1):
        rack_num = tf.concat([rack_num1, rack_num], axis=1)
        p = tf.concat([p, p1], axis=1)
    else:
        rack_num = rack_num1

    if tf.reduce_sum(rack_num) >= Np / 2:
        sum_ = tf.constant(0, dtype=tf.float32)
        for i in range(rack_num.shape[1]):
            sum_ += rack_num[0][i]
            if sum_ == (Np / 2):
                count = i + 1
                break
            if (Np / 2) - 1 < sum_ < (Np / 2):
                sliced_rack_num = rack_num[0, :i + 1]
                # 计算结果
                value_ = (Np / 2) - tf.reduce_sum(sliced_rack_num)
                rack_num = tf.tensor_scatter_nd_update(rack_num, tf.constant([[0, i + 1]]), [value_])
                count = i + 2
                break
        p_new = p[:, :count, :, :]  # 用于填充的的矩阵
        rack_new = rack_num[:, :count]  # 每个矩阵对应的个数

    else:  # 需要的矩阵数大于所有的rack_num之和
        for i in range(rack_num.shape[1]):
            differ = Np / 2 - tf.reduce_sum(rack_num)  # 求出差了多少
            if differ == 0:
                break
            if rack_num[0, i] < differ:
                rack_num_0i = rack_num[0][i] + rack_num[0][i]
                rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
            else:
                d_inter = int(differ)
                d_decimal = differ - int(differ)
                if isinstance(differ, int):
                    rack_num_0i = rack_num[0][i] + differ
                    rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
                else:
                    rack_num_0i = rack_num[0][i] + d_inter
                    rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
                    selected_matrix = p[:, i:i + 1, :, :]
                    # 将选中的矩阵添加在 p 的末尾
                    d_decimal = tf.constant(d_decimal)
                    d_decimal = tf.expand_dims(tf.expand_dims(d_decimal, axis=0), axis=0)
                    p = tf.concat([p, selected_matrix], axis=1)
                    rack_num = tf.concat([rack_num, d_decimal], axis=1)
        p_new = tf.Variable(tf.identity(p))
        rack_new = tf.Variable(tf.identity(rack_num))

    #依次配置
    flag_int = True
    for _, n in enumerate(rack_new[0]):
        n_inter = int(n)
        n_decimal = n - int(n)
        count_num = 0
        while n_inter:
            p_matrix = p_new[0][_]
            oxc_indices = [tuple(sublist) for sublist in
                           tf.where(tf.not_equal(p_matrix, 0)).numpy()]  #找出permutation矩阵中的非零元素并返回索引
            sorted_oxc_indices = [tuple(sorted(pair)) for pair in oxc_indices]
            for connect in sorted_oxc_indices:
                connection_matrix[connect] += 1
            count_num += 1  #用完一个矩阵加一
            if count_num == n_inter:  #当用的矩阵数量等于该矩阵的系数所对应的矩阵个数时跳出，取另一个矩阵，接着循环n次
                break
        if n_decimal:
            flag_int = False
            # 求出目前的连接关系
            trans_matrix = np.transpose(connection_matrix)
            connection_matrix = trans_matrix + connection_matrix
            #check the oxc ports count
            # flag_connect_number = 1
            # for i in range(connection_matrix.shape[0]):
            #     sum_ = sum(connection_matrix[i])
            #     if sum_ != 48:
            #         flag_connect_number = 0
            #判断是否有数据没有连接
            non_zero_indices = (symmetric_matrix > 0) & (connection_matrix == 0)  #查看是否存在有数据但没有连接的情况
            remain_matrix = np.zeros_like(symmetric_matrix)  #存在剩余需求
            remain_matrix[non_zero_indices] = symmetric_matrix[non_zero_indices]  #剩余位置的需求
            remain_matrix = np.triu(remain_matrix)
            for i in range(int(R_S / 2)):
                if np.all(remain_matrix == -1):  #如果没有剩余数据
                    if np.all(copied_matrix == -1):
                        break
                    else:
                        max_P = np.unravel_index(np.argmax(copied_matrix), copied_matrix.shape)  # 将扁平索引转为多维索引
                        row_index, col_index = max_P
                        connection_matrix[row_index][col_index] += 1
                        connection_matrix[col_index][row_index] += 1
                        # 将指定行和列的值赋为0
                        copied_matrix[:, col_index] = -1  # 将列的值赋为0
                        copied_matrix[row_index, :] = -1
                        copied_matrix[col_index, :] = -1  # 将列的值赋为0
                        copied_matrix[:, row_index] = -1
                else:  #如果有剩余值
                    x, y = np.unravel_index(np.argmax(remain_matrix), remain_matrix.shape)
                    if x != y:
                        connection_matrix[x][y] += 1
                        connection_matrix[y][x] += 1
                        remain_matrix[:, y] = -1  # 将列的值赋为0
                        remain_matrix[x, :] = -1
                        remain_matrix[:, x] = -1  # 将列的值赋为0
                        remain_matrix[y, :] = -1
                        copied_matrix[:, y] = -1  # 将列的值赋为0
                        copied_matrix[x, :] = -1
                        copied_matrix[y, :] = -1  # 将列的值赋为0
                        copied_matrix[:, x] = -1
                    else:
                        remain_matrix[x][y] = -1
                        x, y = np.unravel_index(np.argmax(remain_matrix), remain_matrix.shape)
                        connection_matrix[x][y] += 1
                        connection_matrix[y][x] += 1
                        remain_matrix[:, y] = -1  # 将列的值赋为0
                        remain_matrix[x, :] = -1
                        remain_matrix[:, x] = -1  # 将列的值赋为0
                        remain_matrix[y, :] = -1
                        copied_matrix[:, y] = -1  # 将列的值赋为0
                        copied_matrix[x, :] = -1
                        copied_matrix[y, :] = -1  # 将列的值赋为0
                        copied_matrix[:, x] = -1
    if flag_int:
        trans_matrix = np.transpose(connection_matrix)
        connection_matrix = trans_matrix + connection_matrix

    return connection_matrix


# ## 最后一个端口按数据量大的顺序递减分配
# def OXC_connect(matrix, p, c):
#     copied_matrix = np.copy(matrix)
#     p1 = tf.identity(p)
#     connection_matrix = np.zeros((R_S, R_S))
#     rack_num1 = tf.ones_like(c)
#     sum_c = tf.reduce_sum(c) #系数之和
#     inform_c = c / sum_c #系数比例
#     rack_num = tf.round(inform_c * (Np/2)) #需要的矩阵数
#     if Np / 2 > tf.reduce_sum(rack_num1):
#         rack_num = tf.concat([rack_num1, rack_num], axis=1)
#         p = tf.concat([p, p1], axis=1)
#     else:
#         rack_num = rack_num1
#
#     if tf.reduce_sum(rack_num) >= Np / 2:
#         sum_ = tf.constant(0, dtype=tf.float32)
#         for i in range(rack_num.shape[1]):
#             sum_ += rack_num[0][i]
#             if sum_ == (Np / 2):
#                 count = i + 1
#                 break
#             if (Np / 2)-1 < sum_ < (Np / 2):
#                 sliced_rack_num = rack_num[0, :i+1]
#                 # 计算结果
#                 value_ = (Np / 2) - tf.reduce_sum(sliced_rack_num)
#                 rack_num = tf.tensor_scatter_nd_update(rack_num, tf.constant([[0, i+1]]), [value_])
#                 count = i + 2
#                 break
#         p_new = p[:, :count, :, :]  # 用于填充的的矩阵
#         rack_new = rack_num[:, :count]  # 每个矩阵对应的个数
#
#     else:#需要的矩阵数大于所有的rack_num之和
#         differ_ = tf.constant(0, dtype=tf.float32)#用于存放rack_num累计的和，和达到differ则跳出
#         differ = Np / 2 - tf.reduce_sum(rack_num)#求出差了多少
#         for i in range(rack_num.shape[1]):
#             if rack_num[0, i] < differ:
#                 differ_ += rack_num[0][i]
#                 rack_num_0i = rack_num[0][i] + rack_num[0][i]
#                 rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
#             else:
#                 differ_ += differ
#                 rack_num_0i = rack_num[0][i] + differ
#                 rack_num = tf.tensor_scatter_nd_update(rack_num, indices=[[0, i]], updates=[rack_num_0i])
#             if differ_ == differ:
#                 break
#         p_new = tf.Variable(tf.identity(p))
#         rack_new = tf.Variable(tf.identity(rack_num))
#
#     #依次配置
#     for _, n in enumerate(rack_new[0]):
#         n_inter = int(n)
#         n_decimal = n-int(n)
#         count_num = 0
#         while n_inter:
#             p_matrix = p_new[0][_]
#             oxc_indices = [tuple(sublist) for sublist in tf.where(tf.not_equal(p_matrix, 0)).numpy()]#找出permutation矩阵中的非零元素并返回索引
#             sorted_oxc_indices = [tuple(sorted(pair)) for pair in oxc_indices]
#             for connect in sorted_oxc_indices:
#                 connection_matrix[connect] += 1
#             count_num += 1 #用完一个矩阵加一
#             if count_num==n_inter: #当用的矩阵数量等于该矩阵的系数所对应的矩阵个数时跳出，取另一个矩阵，接着循环n次
#                 break
#         if n_decimal:
#             for i in range(int(R_S/2)):
#             # 对值进行排序并记录排序后的索引
#                 max_P = np.unravel_index(np.argmax(copied_matrix), copied_matrix.shape)
#                 row_index, col_index = max_P
#                 # 将指定行和列的值赋为0
#                 copied_matrix[:, col_index] = 0  # 将列的值赋为0
#                 copied_matrix[row_index, :] = 0
#                 copied_matrix[col_index, :] = 0  # 将列的值赋为0
#                 copied_matrix[:, row_index] = 0
#                 connection_matrix[max_P] += 1
#
#     trans_matrix = np.transpose(connection_matrix)
#     connection_matrix = trans_matrix + connection_matrix
#
#     return connection_matrix


def find_paths(matrix, start, end):
    queue = deque([(start, [start])])
    paths = []
    while queue:
        current, path = queue.popleft()
        if current == end:
            paths.append(path)
            continue
        for neighbor, weight in enumerate(matrix[current]):
            if weight > 0 and neighbor not in path:
                if neighbor == end or len(path) < 2:
                    queue.append((neighbor, path + [neighbor]))
    return paths


#多余绕转 sorted order
# def TE(inter_data, oxc_connect_matrix):
#     percent_ = {}
#     B_matrix = oxc_connect_matrix * B  # 每条链路提供的带宽
#     nonzero_indices = np.nonzero(inter_data)
#     # 获取非零元素的值和索引
#     values = inter_data[nonzero_indices]
#     indices = np.column_stack(nonzero_indices)
#     sorted_indices = indices[np.argsort(values)[::-1]]
#     for i in sorted_indices:#取出需要传输的数据
#         sum_ratio = 0
#         rack_n = i[0]
#         rack_m = i[1]
#         link_num = []
#         B_copy = copy.copy(B_matrix[rack_n][rack_m])

#         data = inter_data[rack_n][rack_m] - B_matrix[rack_n][rack_m]
#         if data > 0:#如果大于0则绕转
#             B_matrix[rack_n][rack_m] = 0
#             paths = find_paths(oxc_connect_matrix, rack_n, rack_m)  # 找到n和m间的所有路径
#             for path in paths:
#                 if len(path) == 3:
#                     hop = path[1]
#                     link_num.append(min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]))
#             sum_link_num = sum(link_num)
#             ##根据可用带宽的比例进行分配
#             if paths == []:
#                 if (rack_n, rack_m) not in percent_:
#                     percent_[(rack_n, rack_m)] = []
#                 percent_[(rack_n, rack_m)].append('None')
#             else:
#                 for path in paths:
#                     if len(path) == 2:
#                         if len(paths) == 1:
#                             B_matrix[rack_n][rack_m] -= data
#                             if (rack_n, rack_m) not in percent_:
#                                 percent_[(rack_n, rack_m)] = []
#                             percent_[(rack_n, rack_m)].append(10000)
#                         else:
#                             hop = path[1]
#                             ratio_of_interdata = round(B_copy / inter_data[rack_n][rack_m], 2)
#                             if ratio_of_interdata == 0.0:
#                                 ratio_of_interdata = 0.01
#                             sum_ratio += ratio_of_interdata
#                             if sum_ratio > 1:
#                                 break
#                             else:
#                                 if (rack_n, rack_m) not in percent_:
#                                     percent_[(rack_n, rack_m)] = []
#                                 percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
#                     if len(path) == 3:
#                         hop = path[1]
#                         ratio_of_data = min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]) / sum_link_num
#                         ratio_of_interdata = round(ratio_of_data * data / inter_data[rack_n][rack_m], 2)
#                         if ratio_of_interdata == 0.0:
#                             ratio_of_interdata = 0.01
#                         sum_ratio += ratio_of_interdata
#                         if sum_ratio > 1:
#                             break
#                         else:
#                             B_matrix[rack_n][hop] -= ratio_of_data * data
#                             B_matrix[hop][rack_m] -= ratio_of_data * data
#                             if (rack_n, rack_m) not in percent_:
#                                 percent_[(rack_n, rack_m)] = []
#                             percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
#         else:#否则直连
#             B_matrix[rack_n][rack_m] -= inter_data[rack_n][rack_m]
#             if (rack_n, rack_m) not in percent_:
#                 percent_[(rack_n, rack_m)] = []
#             percent_[(rack_n, rack_m)].append(10000)
#         ##计算所有路径上的可用带宽
#
#     (zero_indices_x, zero_indices_y) = np.where(inter_data == 0)
#
#     for _, (rack_n, rack_m) in enumerate(zip(zero_indices_x, zero_indices_y)):
#         if (rack_n, rack_m) not in percent_:
#             percent_[(rack_n, rack_m)] = []
#         percent_[(rack_n, rack_m)].append(-1)
# # 计算MLU：
#     B_ = oxc_connect_matrix * B - B_matrix
#     MLU_matrix = np.divide(B_, oxc_connect_matrix * B, out=np.full_like(B_, np.nan), where=oxc_connect_matrix * B!=0)
#     min_MLU = np.nanmin(MLU_matrix)
#     max_MLU = np.nanmax(MLU_matrix)
#     mean_MLU = np.nanmean(MLU_matrix)
#     return percent_, max_MLU, mean_MLU


def TE_all(inter_data, oxc_connect_matrix):
    percent_ = {}
    B_matrix = oxc_connect_matrix * B  # 带宽供应矩阵
    nonzero_indices = np.nonzero(inter_data)
    # 获取非零元素的值和索引
    values = inter_data[nonzero_indices]
    indices = np.column_stack(nonzero_indices)
    sorted_indices = indices[np.argsort(values)[::-1]]
    for i in sorted_indices:
        sum_ratio = 0
        rack_n = i[0]
        rack_m = i[1]
        paths = find_paths(oxc_connect_matrix, rack_n, rack_m)  # 找到n和m间的所有路径
        # b_num = []  # 存放这些路径能提供的带宽
        link_num = []
        ##计算所有路径上的可用带宽
        for path in paths:
            if len(path) == 2:
                # b_num.append(B_matrix[rack_n][rack_m])
                link_num.append(oxc_connect_matrix[rack_n][rack_m])
            else:
                hop = path[1]
                # b_num.append(min(B_matrix[rack_n][hop], B_matrix[hop][rack_m]))
                link_num.append(min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]))
        # sum_b_num = sum(b_num)
        sum_link_num = sum(link_num)
        ##根据可用带宽的比例进行分配
        for path in paths:
            if len(path) == 2:
                ratio = oxc_connect_matrix[rack_n][rack_m] / sum_link_num
                # ratio = round(1/19, 2)
                ratio_ofdata = round(ratio, 2)
                if ratio_ofdata == 0.0:
                    ratio_ofdata = 0.01
                sum_ratio += ratio_ofdata
                if sum_ratio > 1:
                    break
                else:
                    B_matrix[rack_n][rack_m] -= ratio * inter_data[rack_n][rack_m]
                    if (rack_n, rack_m) not in percent_:
                        percent_[(rack_n, rack_m)] = []
                    percent_[(rack_n, rack_m)].append([rack_m, ratio_ofdata])
            else:
                hop = path[1]
                # ratio = round(1/19, 2)
                ratio = min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]) / sum_link_num
                # ratio = min(B_matrix[rack_n][hop], B_matrix[hop][rack_m]) / sum_b_num
                ratio_ofdata = round(ratio, 2)
                if ratio_ofdata == 0.0:
                    ratio_ofdata = 0.01
                sum_ratio += ratio_ofdata
                if sum_ratio > 1:
                    break
                else:
                    B_matrix[rack_n][hop] -= ratio * inter_data[rack_n][rack_m]
                    B_matrix[hop][rack_m] -= ratio * inter_data[rack_n][rack_m]
                    if (rack_n, rack_m) not in percent_:
                        percent_[(rack_n, rack_m)] = []
                    percent_[(rack_n, rack_m)].append([hop, ratio_ofdata])

    (zero_indices_x, zero_indices_y) = np.where(inter_data == 0)

    for _, (rack_n, rack_m) in enumerate(zip(zero_indices_x, zero_indices_y)):
        if (rack_n, rack_m) not in percent_:
            percent_[(rack_n, rack_m)] = []
        percent_[(rack_n, rack_m)].append(-1)
    # 计算MLU：
    B_ = oxc_connect_matrix * B - B_matrix
    MLU_matrix = np.divide(B_, oxc_connect_matrix * B, out=np.full_like(B_, np.nan), where=oxc_connect_matrix * B != 0)
    min_MLU = np.nanmin(MLU_matrix)
    max_MLU = np.nanmax(MLU_matrix)
    mean_MLU = np.nanmean(MLU_matrix)

    return percent_, max_MLU, mean_MLU


# non_zeros order  B_sum
def TE_new(inter_data, oxc_connect_matrix):
    percent_ = {}
    B_matrix = oxc_connect_matrix * B  # 每条链路提供的带宽
    nonzero_indices = np.nonzero(inter_data)

    # 获取非零元素的值和索引
    # values = inter_data[nonzero_indices]
    # indices = np.column_stack(nonzero_indices)
    # sorted_indices = indices[np.argsort(values)[::-1]]
    for rack_n, rack_m in zip(nonzero_indices[0], nonzero_indices[1]):  #取出需要传输的数据
        # for rack_n, rack_m in zip(where[0], where[1]):
        # if rack_n == 5:
        #     pdb.set_trace()
        sum_ratio = 0
        # rack_n = i[0]
        # rack_m = i[1]
        link_num = []
        B_num = []
        B_copy = copy.copy(B_matrix[rack_n][rack_m])
        data = inter_data[rack_n][rack_m] - B_matrix[rack_n][rack_m]
        if data > 0:  #如果大于0则绕转
            B_matrix[rack_n][rack_m] = 0
            paths = find_paths(oxc_connect_matrix, rack_n, rack_m)  # 找到n和m间的所有路径
            for path in paths:
                if len(path) == 3:
                    hop = path[1]
                    link_num.append(min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]))
                    B_ = min(B_matrix[rack_n][hop], B_matrix[hop][rack_m])
                    if B_ < 0:
                        B_ = 0
                    B_num.append(B_)
            sum_link_num = sum(link_num)
            sum_B_num = sum(B_num)
            ##根据可用带宽的比例进行分配
            if all(element <= 0 for element in B_num):
                sum_link_num += oxc_connect_matrix[rack_n][rack_m]
                for path in paths:
                    if len(path) == 2:
                        if len(paths) == 1:
                            B_matrix[rack_n][rack_m] -= data
                            if (rack_n, rack_m) not in percent_:
                                percent_[(rack_n, rack_m)] = []
                            percent_[(rack_n, rack_m)].append(10000)
                        else:
                            hop = path[1]
                            ratio_of_data = oxc_connect_matrix[rack_n][rack_m] / sum_link_num
                            ratio_of_interdata = round((B_copy + ratio_of_data * data) / inter_data[rack_n][rack_m], 2)
                            if ratio_of_interdata == 0.0:
                                ratio_of_interdata = 0.01
                            sum_ratio += ratio_of_interdata
                            if sum_ratio > 1:
                                ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                            if ratio_of_interdata == 0.0:
                                break
                            else:
                                if (rack_n, rack_m) not in percent_:
                                    percent_[(rack_n, rack_m)] = []
                                percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
                    if len(path) == 3:
                        hop = path[1]
                        ratio_of_data = min(oxc_connect_matrix[rack_n][hop],
                                            oxc_connect_matrix[hop][rack_m]) / sum_link_num
                        ratio_of_interdata = round(ratio_of_data * data / inter_data[rack_n][rack_m], 2)
                        if ratio_of_interdata == 0.0:
                            ratio_of_interdata = 0.01
                        sum_ratio += ratio_of_interdata
                        if sum_ratio > 1:
                            ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                        if ratio_of_interdata == 0.0:
                            break
                        else:
                            B_matrix[rack_n][hop] -= ratio_of_data * data
                            B_matrix[hop][rack_m] -= ratio_of_data * data
                            if (rack_n, rack_m) not in percent_:
                                percent_[(rack_n, rack_m)] = []
                            percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
            else:
                for path in paths:
                    if len(path) == 2:
                        if len(paths) == 1:
                            B_matrix[rack_n][rack_m] -= data
                            if (rack_n, rack_m) not in percent_:
                                percent_[(rack_n, rack_m)] = []
                            percent_[(rack_n, rack_m)].append(10000)
                        else:
                            hop = path[1]
                            ratio_of_interdata = round(B_copy / inter_data[rack_n][rack_m], 2)
                            # ratio_of_interdata = B_copy / inter_data[rack_n][rack_m]
                            if ratio_of_interdata == 0.0:
                                ratio_of_interdata = 0.01
                            sum_ratio += ratio_of_interdata
                            if sum_ratio > 1:
                                ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                            if ratio_of_interdata == 0.0:
                                break
                            else:
                                if (rack_n, rack_m) not in percent_:
                                    percent_[(rack_n, rack_m)] = []
                                percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
                    if len(path) == 3:
                        hop = path[1]
                        B_check = min(B_matrix[rack_n][hop], B_matrix[hop][rack_m])
                        if B_check <= 0:
                            continue
                        ratio_of_data = B_check / sum_B_num
                        ratio_of_interdata = round(ratio_of_data * data / inter_data[rack_n][rack_m], 2)
                        # ratio_of_interdata = ratio_of_data * data / inter_data[rack_n][rack_m]

                        if ratio_of_interdata == 0.0:
                            ratio_of_interdata = 0.01
                        sum_ratio += ratio_of_interdata
                        if sum_ratio > 1:
                            ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                        if ratio_of_interdata == 0.0:
                            break
                        else:
                            B_matrix[rack_n][hop] -= ratio_of_data * data
                            B_matrix[hop][rack_m] -= ratio_of_data * data
                            if (rack_n, rack_m) not in percent_:
                                percent_[(rack_n, rack_m)] = []
                            percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
        else:  #否则直连
            B_matrix[rack_n][rack_m] -= inter_data[rack_n][rack_m]
            if (rack_n, rack_m) not in percent_:
                percent_[(rack_n, rack_m)] = []
            percent_[(rack_n, rack_m)].append(10000)
        ##计算所有路径上的可用带宽

    (zero_indices_x, zero_indices_y) = np.where(inter_data == 0)

    for _, (rack_n, rack_m) in enumerate(zip(zero_indices_x, zero_indices_y)):
        if (rack_n, rack_m) not in percent_:
            percent_[(rack_n, rack_m)] = []
        percent_[(rack_n, rack_m)].append(-1)
    # 计算MLU：
    B_ = oxc_connect_matrix * B - B_matrix
    MLU_matrix = np.divide(B_, oxc_connect_matrix * B, out=np.full_like(B_, np.nan), where=oxc_connect_matrix * B != 0)
    min_MLU = np.nanmin(MLU_matrix)
    max_MLU = np.nanmax(MLU_matrix)
    mean_MLU = np.nanmean(MLU_matrix)
    return percent_, max_MLU, mean_MLU, MLU_matrix, B_matrix


#original Link_sum
def TE(inter_data, oxc_connect_matrix):
    percent_ = {}
    B_matrix = oxc_connect_matrix * B  # 每条链路提供的带宽
    nonzero_indices = np.nonzero(inter_data)
    # 获取非零元素的值和索引
    # values = inter_data[nonzero_indices]
    # indices = np.column_stack(nonzero_indices)
    # sorted_indices = indices[np.argsort(values)[::-1]]
    for rack_n, rack_m in zip(nonzero_indices[0], nonzero_indices[1]):  #取出需要传输的数据
        # if rack_n == 5:
        #     pdb.set_trace()
        sum_ratio = 0
        # rack_n = i[0]
        # rack_m = i[1]
        link_num = []
        B_copy = copy.copy(B_matrix[rack_n][rack_m])
        data = inter_data[rack_n][rack_m] - B_matrix[rack_n][rack_m]
        if data > 0:  #如果大于0则绕转
            B_matrix[rack_n][rack_m] = 0
            paths = find_paths(oxc_connect_matrix, rack_n, rack_m)  # 找到n和m间的所有路径
            for path in paths:
                if len(path) == 3:
                    hop = path[1]
                    link_num.append(min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]))
            sum_link_num = sum(link_num)
            ##根据可用带宽的比例进行分配
            for path in paths:
                if len(path) == 2:
                    if len(paths) == 1:
                        B_matrix[rack_n][rack_m] -= data
                        if (rack_n, rack_m) not in percent_:
                            percent_[(rack_n, rack_m)] = []
                        percent_[(rack_n, rack_m)].append(10000)
                    else:
                        hop = path[1]
                        ratio_of_interdata = round(B_copy / inter_data[rack_n][rack_m], 2)
                        if ratio_of_interdata == 0.0:
                            ratio_of_interdata = 0.01
                        sum_ratio += ratio_of_interdata
                        if sum_ratio > 1:
                            ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                        if ratio_of_interdata == 0.0:
                            break
                        else:
                            if (rack_n, rack_m) not in percent_:
                                percent_[(rack_n, rack_m)] = []
                            percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
                if len(path) == 3:
                    hop = path[1]
                    ratio_of_data = min(oxc_connect_matrix[rack_n][hop], oxc_connect_matrix[hop][rack_m]) / sum_link_num
                    ratio_of_interdata = round(ratio_of_data * data / inter_data[rack_n][rack_m], 2)
                    if ratio_of_interdata == 0.0:
                        ratio_of_interdata = 0.01
                    sum_ratio += ratio_of_interdata
                    if sum_ratio > 1:
                        ratio_of_interdata = round(1 - (sum_ratio - ratio_of_interdata), 2)
                    if ratio_of_interdata == 0.0:
                        break
                    else:
                        B_matrix[rack_n][hop] -= ratio_of_data * data
                        B_matrix[hop][rack_m] -= ratio_of_data * data
                        if (rack_n, rack_m) not in percent_:
                            percent_[(rack_n, rack_m)] = []
                        percent_[(rack_n, rack_m)].append([hop, ratio_of_interdata])
        else:  #否则直连
            B_matrix[rack_n][rack_m] -= inter_data[rack_n][rack_m]
            if (rack_n, rack_m) not in percent_:
                percent_[(rack_n, rack_m)] = []
            percent_[(rack_n, rack_m)].append(10000)
        ##计算所有路径上的可用带宽

    (zero_indices_x, zero_indices_y) = np.where(inter_data == 0)

    for _, (rack_n, rack_m) in enumerate(zip(zero_indices_x, zero_indices_y)):
        if (rack_n, rack_m) not in percent_:
            percent_[(rack_n, rack_m)] = []
        percent_[(rack_n, rack_m)].append(-1)
    # 计算MLU：
    B_ = oxc_connect_matrix * B - B_matrix
    MLU_matrix = np.divide(B_, oxc_connect_matrix * B, out=np.full_like(B_, np.nan), where=oxc_connect_matrix * B != 0)
    min_MLU = np.nanmin(MLU_matrix)
    max_MLU = np.nanmax(MLU_matrix)
    mean_MLU = np.nanmean(MLU_matrix)
    return percent_, max_MLU, mean_MLU, MLU_matrix


def get_matrix_new(size):
    # 定义元素乘以的数字范围
    # multipliers = [0, 1, 10]
    multipliers1 = [2]
    multipliers2 = [8, 10]
    random_matrix = np.zeros((size, size))
    # 生成随机矩阵
    list_in = list(range(size))
    list_out = list(range(size))
    for i in list_in:
        output = random.choice(list_out)
        while output == i:
            output = random.choice(list_out)
        input = i
        random_matrix[input][output] = np.random.randint(40, 80)
        random_matrix[output][input] = np.random.randint(40, 80)
        list_in.remove(i)
        list_in.remove(output)
        list_out.remove(output)
        list_out.remove(i)

    # random_matrix = np.eye(size)
    # random_matrix = 1 - random_matrix
    # 在每个位置上随机选择一个数字，并与对应位置的随机元素相乘
    for i in range(size):
        for j in range(size):
            if 40 <= random_matrix[i, j] <= 60:
                random_multiplier = np.random.choice(multipliers1)
                random_matrix[i, j] *= random_multiplier
                random_matrix[i, j] *= 2
            else:
                random_multiplier = np.random.choice(multipliers2)
                random_matrix[i, j] *= random_multiplier
                random_matrix[i, j] *= 2
    return random_matrix


def get_matrix(size):
    # 定义元素乘以的数字范围
    # multipliers = [0, 1, 10]
    multipliers1 = [0]
    multipliers2 = [8, 10]
    random_matrix = np.random.randint(30, 60, (size, size))

    # random_matrix = np.eye(size)
    # random_matrix = 1 - random_matrix
    # 在每个位置上随机选择一个数字，并与对应位置的随机元素相乘
    for i in range(size):
        for j in range(size):
            if 30 <= random_matrix[i, j] <= 50:
                random_multiplier = np.random.choice(multipliers1)
                random_matrix[i, j] *= random_multiplier
                random_matrix[i, j] *= 2
            else:
                random_multiplier = np.random.choice(multipliers2)
                random_matrix[i, j] *= random_multiplier
                random_matrix[i, j] *= 2
            if i == j:
                random_matrix[i, j] = 0
    return random_matrix


def format_nested_list(data):
    if isinstance(data, list):
        formatted_sublists = [format_nested_list(sublist) for sublist in data]
        return '[' + ','.join(formatted_sublists) + ']'
    else:
        return str(data)


if __name__ == "__main__":
    R_S = 256
    Np = 255  ##
    B = 100
    # matrix = np.array([[0, 480, 180, 480, 50],
    #                    [75, 0, 75, 75, 75],
    #                    [480, 80, 0, 180, 220],
    #                    [75, 50, 220, 0, 200],
    #                    [480, 20, 100, 360, 0]])
    # matrix = get_matrix_new(R_S)
    # matrix = get_matrix(R_S)
    # np.savetxt('128_x2_new/bvn/data_matrix_30_60.txt', matrix, fmt='%d')
    matrix = np.loadtxt('256/bvn/data_matrix_30_60_new.txt', dtype=int)
    # print("生成的矩阵:", matrix)
    #计算每行和每列的和
    # row_sums = np.sum(matrix, axis=1)
    # col_sums = np.sum(matrix, axis=0)
    # # 打印结果
    # print("每行的和:")
    # print(row_sums)
    # print("\n每列的和:")
    # print(col_sums)
    # ##data_matrix_1
    # matrix = np.array([[0, 430, 5, 25, 0],
    #                    [49, 0, 0, 0, 0],
    #                    [180, 65, 0, 610, 0],
    #                    [83, 34, 720, 0, 5],
    #                    [73, 190, 850, 10, 0]])
    #data_matrix_2
    # matrix = np.array( [[0, 1920, 720, 200, 200],
    #                  [300, 0, 300, 300, 300],
    #                  [720, 320, 0, 720, 880],
    #                  [200, 200, 880, 0,  800],
    #                  [100, 80, 400, 1440, 0]])
    #data_matrix_3
    # matrix = np.array([[0, 2000, 720, 200, 200],
    #                    [100, 0, 300, 300, 800],
    #                    [720, 320, 0, 720, 1200],
    #                    [200, 200, 880, 0, 800],
    #                    [300, 80, 400, 1440, 0]])
    # data_matrix_4
    # matrix = np.array([[0, 4000, 100, 0, 50],
    #                    [0, 0, 150, 50, 3000],
    #                    [3000, 0, 0, 40, 100],
    #                    [200, 200, 2500, 0, 400],
    #                    [100, 80, 0, 3000, 0]])

    # matrix = np.array([[0, 4000, 100, 0, 50],
    #                    [0, 0, 150, 50, 3000],
    #                    [3000, 0, 0, 40, 100],
    #                    [200, 200, 2500, 0, 400],
    #                    [100, 80, 0, 3000, 0]])
    # inter_rack_ = np.array([[0, 120, 2800, 0, 500],
    #  [800, 0, 100, 0, 3000],
    #  [180, 0, 0, 500, 200],
    #  [0, 750, 1500, 0, 200],
    #  [100, 0, 100, 500, 0]])
    # inter_rack_ = np.random.randint(0, 201, size=(R_S, R_S))
    # # # # #
    # matrix = np.array([[0, 480, 180, 50, 50],
    #                    [75, 0, 75, 75, 75],
    #                    [180, 80, 0, 180, 220],
    #                    [50, 50, 220, 0, 200],
    #                    [25, 20, 100, 360, 0]])

    # matrix *= 4
    # matrix = np.array([[0, 4000, 100, 50, 50],
    #                    [50, 0, 150, 50, 3000],
    #                    [3000, 120, 0, 40, 100],
    #                    [200, 200, 2500, 0, 800],
    #                    [100, 80, 50, 3000, 0]])

    # [0, 50, 0, 50, 0]])
    # inter_rack_ = np.array( [[0, 0, 2800, 0, 500],
    #  [800, 0, 0 , 0, 0],
    #  [180, 0, 0, 0, 200],
    #  [0, 0, 0, 0,  1500],
    #  [0, 100, 0, 200 ,0]])

    # inter_rack_ = np.array([[0, 0, 1400, 0, 200],
    #  [800, 0, 0 , 0, 0],
    #  [180, 0, 0, 0, 100],
    #  [0, 0, 0, 0,  800],
    #  [0, 100, 0, 100 ,0]])

    # # 将对角线上的值置为零
    # np.fill_diagonal(inter_rack_, 0)
    # 保存为txt格式

    # ## 重配的例子
    # inter_rack_ = np.array([[0, 0, 0, 0],
    #                         [0, 0, 292, 80],
    #                         [0, 0, 0, 0],
    #                         [28, 0, 0, 0]])

    ## 多跳的例子:
    # inter_rack_ = np.array([[0,   0,   0,   0],
    #                         [83,   0, 117,  34],
    #                         [83,   0,   0, 154],
    #                         [83,   0, 141,  0]])

    # inter_rack_ = np.array([
    #     [0, 312, 0, 0],
    #     [93, 0, 0, 0],
    #     [66, 0, 0, 0],
    #     [54, 130, 0, 0]
    # ])
    start_time_4BvN = time.time()
    inter_symmetric = make_symmetric(matrix)
    print('对称后的流量矩阵：', inter_symmetric)

    ## 选择效果最好的填充方式
    D_stuffing1 = stuffing_min(inter_symmetric)

    flag2 = False
    flag3 = False
    if not np.all(np.diag(D_stuffing1 == 0)):
        # 随机选择要执行的函数
        flag2 = True
        chosen_function = random.choice([stuffing_new_min, stuffing_max])
        D_stuffing2 = chosen_function(inter_symmetric)

        # 执行完函数后再进行另一个条件判断
        if not np.all(np.diag(np.diag(D_stuffing2 == 0))):
            # 如果满足另一个条件，则执行另一个函数
            flag3 = True
            if chosen_function == stuffing_new_min:
                D_stuffing3 = stuffing_max(inter_symmetric)
            else:
                D_stuffing3 = stuffing_new_min(inter_symmetric)

    if flag2 and flag3:
        diag_sum1 = np.trace(D_stuffing1)
        diag_sum2 = np.trace(D_stuffing2)
        diag_sum3 = np.trace(D_stuffing3)
        if diag_sum1 <= diag_sum2 and diag_sum1 <= diag_sum3:
            D_stuffing = D_stuffing1
        elif diag_sum2 <= diag_sum3 and diag_sum2 <= diag_sum3:
            D_stuffing = D_stuffing2
        else:
            D_stuffing = D_stuffing3
    elif flag2 and not flag3:
        diag_sum1 = np.trace(D_stuffing1)
        diag_sum2 = np.trace(D_stuffing2)
        if diag_sum1 <= diag_sum2:
            D_stuffing = D_stuffing1
        elif diag_sum2 <= diag_sum1:
            D_stuffing = D_stuffing2

    else:
        D_stuffing = D_stuffing1

    print('填充后的矩阵:', D_stuffing)

    print('填充后对角线的值：', np.trace(D_stuffing))  #确保无对角线填充
    #
    ## 分解
    p, c = Bvn_composition(D_stuffing)
    print("Permutation_matrix:", p)
    print("coefficient:", c)
    end_time_4BvN = time.time()
    print("BvN time for 256pod is", end_time_4BvN - start_time_4BvN)
    # count_ones = tf.reduce_sum(tf.cast(tf.equal(c, 1), tf.int32))# 检查分解系数中1的个数
    ## 配置
    # oxc_connection = OXC_connect_all(matrix, p, c)
    # oxc_connection = OXC_connect_proportion(matrix, p, c)
    #
    # print("OXC配置方案：", oxc_connection)
    # #计算多跳和MLU
    # percent_, MLU, mean_MLU, MLU_matrix, B_matrix = TE_new(matrix, oxc_connection)
    # # percent_, MLU, mean_MLU, MLU_matrix = TE(matrix, oxc_connection)
    # # where = np.where(MLU_matrix == 1)
    # # oxc_ = oxc_connection[where]
    # # B_ = B_matrix[where]
    # # matrix_ = matrix[where]
    #
    # #输出最大的MLU
    # print("最大的MLU是{}".format(MLU))
    # print('mean_MLU', mean_MLU)
    # # where = np.where(MLU_matrix == MLU)
    # # print('percent:', percent_)
    #
    # end_time = time.time()
    # all_time = end_time - start_time
    # print("程序运行时间：", all_time)
    # np.savetxt('5/bvn/oxc_configuration_5.txt', oxc_connection, fmt='%d', encoding='utf-8')
    # #
    # # with open('percent4check.txt', 'w') as file:
    # #     for key, value in percent_.items():
    # #         line = f"{key}: {', '.join(map(str, value))}\n"
    # #         file.write(line)
    # #
    # with open('5/bvn/data_percent_5.txt', 'w') as f:
    #     keys = sorted(percent_.keys())
    #     current_row = keys[0][0]
    #     for i in range(len(keys)):
    #         if keys[i][0] != current_row:
    #             f.write('\n')
    #             current_row = keys[i][0]
    #         value = percent_[keys[i]]
    #         if value[0] == -1 or value[0] == 10000:
    #             values = str(value)
    #         else:
    #             value.insert(0, len(value))
    #             values = formatted_data = format_nested_list(value)
    #         f.write(values + ' ')
