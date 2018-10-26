import sys
#import urllib2
import numpy as np

#Download data
# url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_test.dat'
# f = urllib2.urlopen(url)
# with open("hw4_test.dat", "wb") as code:
#    code.write(f.read())
#
# url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_train.dat'
# f = urllib2.urlopen(url)
# with open("hw4_train.dat", "wb") as code:
#    code.write(f.read())

#load data
def load_data(filename):
    code = open(filename, "r")
    lines = code.readlines()
    xn = np.zeros((len(lines), 3)).astype(np.float)
    yn = np.zeros((len(lines),)).astype(np.int)

    for i in range(0, len(lines)):
        line = lines[i]
        line = line.rstrip('\r\n').replace('\t', ' ').split(' ')
        xn[i, 0] = 1
        for j in range(1, len(xn[0])):
            xn[i, j] = float(line[j-1])
        yn[i] = int(line[len(xn[0]) - 1])
    return xn, yn

def calculate_W_reg(x, y, lambda_value):
    return  np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x)+lambda_value * np.eye(x.shape[1])), x.transpose()), y)

# test result
def calculate_Eout_or_Ein(w, x, y):
    scores = np.dot(w, x.transpose())
    predicts = np.where(scores>=0, 1.0, -1.0)
    Eout = sum(predicts != y)
    return (Eout*1.0) / predicts.shape[0]

if __name__ == '__main__':
    # prepare train and test data
    train_x, train_y = load_data("hw4_train.dat")
    test_x,test_y = load_data("hw4_test.dat")

    #Q13
    lambda_value = 10
    W = calculate_W_reg(train_x, train_y, lambda_value)
    Ein = calculate_Eout_or_Ein(W, train_x, train_y)
    Eout = calculate_Eout_or_Ein(W, test_x, test_y)
    print ('Q13: Ein = ', Ein, ', Eout= ', Eout)

    #Q14-Q15
    Ein_min = sys.maxint
    optimal_Eout = 0
    optimal_lambda_Ein = 0

    Eout_min = sys.maxint
    optimal_Ein = 0
    optimal_lambda_Eout = 0
    for lambda_value in range(2, -11, -1):
        # calculate ridge regression W
        w_reg = calculate_W_reg(train_x, train_y, pow(10,lambda_value))
        Ein = calculate_Eout_or_Ein(w_reg, train_x, train_y)
        Eout = calculate_Eout_or_Ein(w_reg, test_x, test_y)
        # update Ein Eout lambda
        if Ein_min > Ein:
            Ein_min = Ein
            optimal_lambda_Ein = lambda_value
            optimal_Eout = Eout

        if Eout_min > Eout:
            Eout_min = Eout
            optimal_lambda_Eout = lambda_value
            optimal_Ein = Ein
    #Q14
    print ('Q14: log10 = ', optimal_lambda_Ein, ', Ein= ', Ein_min, ', Eout = ', optimal_Eout)
    #Q15
    print ('Q15: log10 = ', optimal_lambda_Eout, ', Ein = ', optimal_Ein, ', Eout= ', Eout_min)

    # Q16-Q18
    Etrain_min = sys.maxint
    Eval_min = sys.maxint
    Eout_Etrain_min = 0
    Eout_Eval_min = 0
    Eval_Etrain_min = 0
    Etrain_Eval_min = 0
    optimal_lambda_Etrain_min = 0
    optimal_lambda_Eval_min = 0
    split = 120
    for lambda_value in range(2, -11, -1):
        w_reg = calculate_W_reg(train_x[:split], train_y[:split], pow(10, lambda_value))
        Etrain = calculate_Eout_or_Ein(w_reg, train_x[:split], train_y[:split])
        Eval = calculate_Eout_or_Ein(w_reg, train_x[split:],train_y[split:])
        Eout = calculate_Eout_or_Ein(w_reg, test_x, test_y)

        if Etrain_min > Etrain:
            optimal_lambda_Etrain_min = lambda_value
            Etrain_min = Etrain
            Eout_Etrain_min = Eout
            Eval_Etrain_min = Eval

        if Eval_min > Eval:
            optimal_lambda_Eval_min = lambda_value
            Eout_Eval_min = Eout
            Eval_min = Eval
            Etrain_Eval_min = Etrain
    #Q16
    print ('Q16: log10 = ', optimal_lambda_Etrain_min, ', Etrain= ', Etrain_min, ', Eval = ', Eval_Etrain_min, ', Eout = ', Eout_Etrain_min)
    #Q17
    print ('Q17: log10 = ', optimal_lambda_Eval_min, ', Etrain= ', Etrain_Eval_min, ', Eval = ', Eval_min, ', Eout = ', Eout_Eval_min)

    #Q18
    w_reg = calculate_W_reg(train_x, train_y, pow(10, optimal_lambda_Eval_min))
    optimal_Ein = calculate_Eout_or_Ein(w_reg, train_x, train_y)
    optimal_Eout = calculate_Eout_or_Ein(w_reg, test_x, test_y)
    print ('Q18: Ein = ', optimal_Ein, ', Eout = ', optimal_Eout)

    #Q19
    folder_num = 5
    split_folder = 40

    Ecv_min = sys.maxint
    optimal_lambda = 0
    for lambda_value in range(2, -11, -1):
        total_cv = 0
        for i in range(folder_num):
            test_data_x = train_x[i * split_folder:(i + 1) * split_folder, :]
            test_data_y = train_y[i * split_folder:(i + 1) * split_folder]
            if i > 0 and i < (folder_num - 1):
                train_data_x = np.concatenate((train_x[0:i * split_folder, :], train_x[(i + 1) * split_folder:, :]), axis=0)
                train_data_y = np.concatenate((train_y[0:i * split_folder], train_y[(i + 1) * split_folder:]), axis=0)
            elif i == 0:
                train_data_x = train_x[split_folder:, :]
                train_data_y = train_y[split_folder:]
            else:
                train_data_x = train_x[0:i * split_folder, :]
                train_data_y = train_y[0:i * split_folder]

            w_reg = calculate_W_reg(train_data_x, train_data_y, pow(10, lambda_value))
            Ecv = calculate_Eout_or_Ein(w_reg, test_data_x, test_data_y)
            total_cv += Ecv
        total_cv = total_cv * 1. / folder_num
        if Ecv_min > total_cv :
            Ecv_min = total_cv
            optimal_lambda = lambda_value

    print ('Q19: log10=', optimal_lambda, ' Ecv=', Ecv_min)

    w_reg = calculate_W_reg(train_x, train_y, pow(10, optimal_lambda))
    Ein = calculate_Eout_or_Ein(w_reg, train_x, train_y)
    Eout = calculate_Eout_or_Ein(w_reg, test_x, test_y)
    print ('Q20: Ein = ', Ein, 'Eout = ', Eout)



