import re
import sys
import csv
import math
import numpy as np

def parse_results(input_file, coeffi):
    print("predicting values")
    input_f = open(input_file, 'r')
    conv_signs = ['conv', 'res','cccp']
    fc_signs = ['ip','fc','innerproduct']
    pool_signs = ['pool']
    drop_signs = ['drop']
    concat_signs = ['concat']
    inputs = []
    total_power_usage = 0.0
    print(f"input_f length: {len(input_file)}")
    for line in input_f:
        print(f"line: {line}")
        if len(line.strip()) == 0: continue
        if 'json' in line and 'Network' in line.split()[0]:
            print("\n%s", line.split('/')[-1])
        layer_name = line.split()[0].lower()
        print(f"layer name: {layer_name}")
        outputs = re.findall(r"\[(.*?)\]", line)
        # outputs = list(map(float, outputs.split(',')))
        print(outputs)
        if any(conv in layer_name for conv in conv_signs):
            # print("found convolution layer")
            res = []
            items = line.split(':')
            # print(f"item len for conv layer: {len(items)}")
            # print(f"items for conv layer: {items}")
            if len(items) == 5:
                outputs = list(map(float, outputs[0].split(',')))
                print(f"found conv outputs: {outputs}")

                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:] #Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1]) #Filters
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2]) #Padding
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3]) #strides
                # res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5]) #Inputs
                # res += "".join(str(x) for x in inputs)
                res += inputs
                print(f"found items for conv layer: {res}")
                inputs = outputs
                input = list(map(float, res))
                pre_runtime, pre_power = predict_runtime_power('conv', input, coeffi)
                print("%s\t%.3f\t%.3f", layer_name, pre_runtime, pre_power)
                total_power_usage += pre_runtime * pre_power
        if any(fc in layer_name for fc in fc_signs):
            res = []
            items = line.split(':')
            if len(items) == 4:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-2:] #Output
                res += ['1', '1'] # paddings
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])[-1:] #Filters
            if len(items) == 6:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4::3] #Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])[-3:] #Inputs

                input = map(float, res)
                pre_runtime, pre_power = predict_runtime_power('fc', input, coeffi)
                print("%s\t%.3f\t%.3f", layer_name, pre_runtime, pre_power)
        if any(pool in layer_name for pool in pool_signs):
            res = []
            items = line.split(':')
            if len(items) == 3:
                outputs = list(map(float, outputs[0].split(',')))
                print(f"found pool outputs: {outputs}")

                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])[-2:]  # Kernel
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2])[-2:]  # Stride
                # res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])[-3:-1]  # Input
                res += inputs
                input = list(map(float, res))
                inputs = outputs
                pre_runtime, pre_power = predict_runtime_power('pool', input, coeffi)
                print("%s\t%.3f\t%.3f", layer_name, pre_runtime, pre_power)
                total_power_usage += pre_runtime * pre_power

        if any(drop in layer_name for drop in drop_signs):
            res = []
            items = line.split(':')
            if len(items) == 3:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])  # Prob
                tmp = re.findall(r"[-+]?\d*\.\d+|\d+", items[2]) #input
                if len(tmp) == 4:
                    res += tmp
                elif len(tmp) == 2:
                    res += [tmp[0], '1', '1', tmp[1]]
                else:
                    print("Dropout layer parsing error!!")

                input = list(map(float, res))
                pre_runtime, pre_power = predict_runtime_power('drop', input, coeffi)
                print("%s\t%.3f\t%.3f", layer_name, pre_runtime, pre_power)
                total_power_usage += pre_runtime * pre_power

        if any(concat in layer_name for concat in concat_signs):
            res = []
            items = line.split(':')
            if len(items) == 2:
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
                res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])[3::4]  # input, only last dimension
                res = map(float, res)
                while len(res) < 10:
                        res.append(0)
                input = map(float, res)
                pre_runtime, pre_power = predict_runtime_power('concat', input, coeffi)
                print("%s\t%.3f\t%.3f", layer_name, pre_runtime, pre_power)

        if len(outputs) >= 1 and layer_name == "data":
            inputs = list(map(float, outputs[0].split(','))) # this is for the input layer case
    input_f.close()
    return total_power_usage

def predict_runtime_power(type, input, coeffi):
    if type == 'conv':
        input_1 = input[0:3] + input[5:8] + input[13:14]
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i]*input_1[j])
        input_3 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                for k in range(j, len(input_1)):
                    input_3.append(input_1[i]*input_1[j]*input_1[k])
        input_1log = input_1 + list(map(np.log2, input_1))
        input_2log = []
        for i in range(len(input_1log)):
            for j in range(i, len(input_1log)):
                input_2log.append(input_1log[i]*input_1log[j])
        p = input
        input_others = [p[1] * p[2] * p[3] * p[4] * p[5] * p[6], #output pixels
                        p[12] * p[1] * p[2] * p[3] * p[4] * p[5] * p[6],
                        p[12] * p[1] * p[2] * p[4] * p[5] * p[6],
                        p[0] * p[1] * p[2] * p[3], #output data
                        p[4] * p[5] * p[6] * p[7], #filter data
                        p[12] * p[13] * p[14] * p[15], #input data
                        p[12] * p[14] * p[15], #input data
                        p[12] * p[13] * p[15]] #input data
        input_runtime = input_1 + input_2 + input_3 \
                    + input_others + [1] # 1 is the intercept
        input_power = input_1log + input_2log \
                    + input_others + [1] # 1 is the intercept
        runtime = max(sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime'])), 0.105)
        power = sum(np.array(input_power) * np.array(coeffi[type, 'power']))
        return runtime, power
    if type == 'fc':
        input_1 = input
        input_1log = input_1 + list(map(np.log2, input_1))
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i]*input_1[j])
        input_2log = []
        for i in range(len(input_1log)):
            for j in range(i, len(input_1log)):
                input_2log.append(input_1log[i]*input_1log[j])
        p = input
        input_others = [p[0] * p[1] * p[2] * p[3] * p[4] #operations pixels
                        ]
        input_runtime = input_1 + input_2 + input_others + [1] # 1 is the intercept
        input_power = input_1log + input_2log + [1] # 1 is the intercept
        runtime = max(sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime'])), 0.105)
        power = sum(np.array(input_power) * np.array(coeffi[type, 'power']))
        return runtime, power
    if type == 'pool':
        input_1 = input[0:5] + input[8:9]
        input_1log = input_1 + list(map(np.log2, input_1))
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i]*input_1[j])
        input_2log = []
        for i in range(len(input_1log)):
            for j in range(i, len(input_1log)):
                input_2log.append(input_1log[i]*input_1log[j])
        input_3 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                for k in range(j, len(input_1)):
                    input_3.append(input_1[i]*input_1[j]*input_1[k])
        p = input
        input_others = [p[0] * p[1] * p[2] * p[3] * p[4] * p[5], #operations pixels
                        p[0] * p[1] * p[2] * p[3], #output data
                        p[0] * p[8] * p[9] * p[3] #input data
                         ] #input data
        input_runtime = input_1 + input_2 + input_3 \
                    + input_others + [1] # 1 is the intercept
        input_power = input_1log + input_2log + input_others + [1] # 1 is the intercept
        runtime = max(sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime'])), 0.105)
        power = sum(np.array(input_power) * np.array(coeffi[type, 'power']))
        return runtime, power
    if type == 'concat':
        input_1 = input
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i]*input_1[j])
        p = input
        input_others = [p[0] * p[1] * p[2] * p[3], #operations pixels
                        p[0] * p[1] * p[2] * p[4], #input data
                        p[0] * p[1] * p[2] * p[5], #input data
                        p[0] * p[1] * p[2] * p[6], #input data
                        p[0] * p[1] * p[2] * p[7], #input data
                         ] #input data
        input_runtime = input_1 + input_others + [1] # 1 is the intercept
        input_power = input_1 + input_2 \
                      + input_others + [1] # 1 is the intercept
        runtime = max(sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime'])), 0.105)
        power = sum(np.array(input_power) * np.array(coeffi[type, 'power']))
        return runtime, power
    if type == 'drop':
        input_1 = input
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i]*input_1[j])
        p = input
        input_others = [p[0] * p[1] * p[2] * p[3] * p[4], #operations pixels
                        p[0] * p[1] * p[2] * p[3] * p[4] #output data
                         ] #input data
        input_all = input_1 + input_others + [1] # 1 is the intercept
        runtime = sum(np.array(input_all) * np.array(coeffi[type, 'runtime']))
        power = sum(np.array(input_all) * np.array(coeffi[type, 'power']))
        return runtime, power

def parse_coeff(coeffi):
    with open('coeff_conv.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('conv', 'runtime')] = list(map(float, next(res)))
        coeffi[('conv', 'power')] = list(map(float, next(res)))
    # with open('coeff_fc.txt', 'r') as f:
    #     res = csv.reader(f)
    #     coeffi[('fc', 'runtime')] = map(float, next(res))
    #     coeffi[('fc', 'power')] = map(float, next(res))
    with open('coeff_pool.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('pool', 'runtime')] = list(map(float, next(res)))
        coeffi[('pool', 'power')] = list(map(float, next(res)))
    with open('coeff_drop.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('drop', 'runtime')] = list(map(float, next(res)))
        coeffi[('drop', 'power')] = list(map(float, next(res)))
    # with open('coeff_concat.txt', 'r') as f:
    #     res = csv.reader(f)
    #     coeffi[('concat', 'runtime')] = map(float, next(res))
    #     coeffi[('concat', 'power')] = map(float, next(res))
    return coeffi

if __name__ == '__main__':
    coeffi = {}
    parse_coeff(coeffi)
    total_power_usage = parse_results(sys.argv[1], coeffi)
    print(f"total energy usage: {total_power_usage} joules")
    '''
    if len(sys.argv) > 2:
        parse_results(sys.argv[1] #input file
                  , sys.argv[2]) #output_initial
    else:
        parse_results(sys.argv[1])
    '''
