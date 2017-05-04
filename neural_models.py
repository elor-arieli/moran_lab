__author__ = 'elor'

import yaml
import os
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
from random import randint,uniform
from scipy.special import gammaln
from math import log,factorial

def load_params(file):
    with open(file,'r') as param_file:
        params = yaml.safe_load(param_file)
    return params

def get_Alpha_Beta_Gamma(num_trials,num_states,time_points):
    alphas = zeros((num_states,num_trials,time_points))
    betas = zeros((num_states, num_trials, time_points))
    gammas = zeros((num_states, num_trials, time_points))
    return (alphas,betas,gammas)

def get_Aij_matrix(model,states,diag):
    Aij = np.zeros((states,states))

    if model == 'FX':
        for i in range(states):
            for j in range(states):
                if i == j:
                    Aij[i,j] = diag
                elif j == i+1:
                    Aij[i,j] = 1 - diag
        Aij[-1,-1] = 1

    elif model == 'F':
        for i in range(states):
            for j in range(states):
                if i == j:
                    Aij[i,j] = diag
                elif j == i + 1:
                    Aij[i,j] = (1 - diag)/sum(range(states-i))*(states-j+1)
        Aij[-1, -1] = 1

    elif model == 'FShen':
        Aij = np.asmatrix([[3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                           [3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                           [3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                           [3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                           [3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                           [0, 0, 0, 0, 0, 2, 4, 4, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        for i in range(states):
            if Aij[i,0] == 3:
                count_threes += 1
        for i in range(states):
            for j in range(states):
                if Aij[i,j] == 2:
                    Aij[i,j] = diag
                elif Aij[i,j] == 3:
                    Aij[i,j] = 1 / count_threes
                elif Aij[i,j] == 4:
                    Aij[i,j] = (1 - diag) / 2
                elif Aij[i,j] == 5:
                    Aij[i,j] = 1 - diag

    else:
        for i in range(states):
            for j in range(states):
                if i == j:
                    Aij[i,j] = diag
                elif j == i + 1:
                    Aij[i,j] = (1 - diag) / (states - 1)
    return Aij

def get_Bvk_matrix(spike_matrix, states):
    # creates the B matrix of firing probabilities and gives each neuron in each state a rate based on a 10 bin average
    Bvk = np.zeros((states,len(spike_matrix[0])))
    for state in range(states):
        for neuron in range(len(spike_matrix[0])):
            average = np.mean(spike_matrix[randint(0,len(spike_matrix)-1)][neuron][1:10])
            rate = average + uniform(-0.1*average,0.1*average)
            Bvk[state,neuron] = rate
    return Bvk

def fill_alpha_beta_gamma(alphas,bettas,gammas,Aij,Bvk,spike_matrix):
    """
    :param alphas:
    :param bettas:
    :param gammas:
    :param Aij: the transition matrix
    :param Bvk: the firing rate probabilities matrix
    :param spike_matrix: the matrix of neuronal firing rates per trial and bin: matrix[trial][neuron][time_bin]
    :return: returns the filled in alphas, bettas and gammas tables while going forward to calculate them with the initialization and induction formulas from rabiner
    """
    alphas[0,:,0] = 1
    bettas[:,:,-1] = 1
    for tri in range(len(alphas[0])):
        print 'filling alpha tables'
        for t in range(1,len(alphas[0][0])):
            # fill in alphas - posterior probability
            for new_state in range(len(alphas)):
                prob = 0
                for old_state in range(len(alphas)):
                    prob += alphas[old_state,tri,t-1]*Aij[old_state,new_state]
                poisson_prob_sum = 0
                for neuron in range(len(spike_matrix[0])):
                    poisson_prob_sum += poisson_probability(spike_matrix[tri,neuron,t],Bvk[new_state,neuron])
                prob *= poisson_prob_sum
                alphas[new_state,tri,t] = prob
        print 'filling betta tables'
        for t in range(len(alphas[0][0])-2,-1,-1):
            # fill in bettas - anterior probability
            for old_state in range(len(alphas)):
                prob = 0
                for new_state in range(len(alphas)):
                    poisson_prob_sum = 0
                    for neuron in range(len(spike_matrix[0])):
                        poisson_prob_sum += poisson_probability(spike_matrix[tri, neuron, t + 1],Bvk[new_state, neuron])
                    prob += Aij[old_state,new_state]*poisson_prob_sum*bettas[new_state,tri,t+1]
                bettas[old_state,tri,t] = prob
        print 'filling gamma tables'
        for t in range(0, len(alphas[0][0])):
            # fill in gammas - anterior and posterior probability
            sum_alpha_betta = 0
            for state in range(len(alphas)):
                sum_alpha_betta += alphas[state,tri,t]*bettas[state,tri,t]
            for state in range(len(alphas)):
                gammas[state,tri,t] = alphas[state,tri,t]*bettas[state,tri,t]/sum_alpha_betta
    return (alphas,bettas,gammas)

def poisson_probability(rate, Bvk_poisson_mean):
    prob = (Bvk_poisson_mean**rate)*np.exp(-1*Bvk_poisson_mean)/factorial(rate)
    return prob

def run_HMM(directory, index, spike_matrix):
    os.system('cd ' + directory)

    print 'loading parameters and setting directories'

    #load params into local variables
    params = load_params(str(index) + '.yaml')['params']
    allcounts = params['allcounts']
    threshold = params['params']['threshold']
    newdirname = params['newdirname']
    diag = params['diag']
    states = params['states']
    model = params['model']
    neurons = params['neurons']
    binWpoiss = params['binWpoiss']
    its = params['its']

    # go the new dir
    os.system('md ' + newdirname)
    os.system('cd ' + newdirname)

    TrialsNum, unitsNum, T = np.shape(spike_matrix)
    TrialsNum, unitsNum, T = int(TrialsNum), int(unitsNum), int(T)

    print 'creating Aij and Bk matrixes'
    Aij = get_Aij_matrix(model,states,diag)
    Bvk = get_Bvk_matrix(spike_matrix)

    # cm = count.max()
    # if cm > 50000:
    #     dnorm = gammaln(count +1)
    # else:
    #     """
    #     what?
    #     original:
    #     tmp = cumsum([0; log((1:max(max(max(count)))).')]);
    #     """
    #     tmp = np.cumsum([0:log(np.max(count))])
    #     dnorm = tmp(count+1)

    print 'computing variables'
    log_l = zeros((1,its))
    alphas,betas,gammas = get_Alpha_Beta_Gamma(TrialsNum,states,T)
    scale = np.matlib.repmat([1 zeros(1,T-1)],TrialsNum ,1)
    xsi = zeros(TrialsNum,T, numStates, numStates)

    print 'running iterations'
    for itr in range(its):
        if itr % 10 == 0:
            print 'Iteration: %s' % itr
        log_l_tri = zeros((1,TrialsNum))
        for tri in range(TrialsNum):
            countTr = np.squeeze(count[tri,:,:])
            dnormTr = np.squeeze(dnorm[tri,:,:])
            alphas[tri,:,:],betas[tri,:,:],gammas[tri,:,:],xsi[tri,:,:,:],log_l_tri[tri] = hmmdecodePoiss2(countTr,dnormTr,Aij,rate)

        log_l[itr] = -sum(log_l_tri)
        conver = 1000
        if itr > 1:
            conver = (log_l[itr-1]-log_l[itr])/log_l[itr]
        if itr > 1 and conver < threshold:
            print 'converged on iteration num: %s' % itr
            break

    for i in range(numStates):
        for j in range(numStates):
            Aij[i,j] =





# what is this??
    # if (strcmp(version('-release'), '2007b') == 1)
    #     rand('twister', sum(100*clock));
    # else
    #     RandStream.setDefaultStream ...
    #      (RandStream('mt19937ar','seed',sum(100*clock)));
    # end


    for i=1:numStates
        for j=1:numStates
            Aij(i,j) = sum((sum(xsi(:,:,i,j))))/sum(sum(gammas(:,1:T-1,i)));
        end
    end

    rate = zeros(size(rate));
    normRate = zeros(size(rate));

    for u=1:unitsNum
        for i=1:numStates
            for tri=1:TrialsNum
                for t=1:T-1
                    rate(u,i) = rate(u,i) + squeeze(gammas(tri,t,i))*count(tri,t+1,u);
                end
                normRate(u,i) = normRate(u,i) + sum(squeeze(gammas(tri,:,i)));
            end
            if normRate(u,i) == 0
                rate(u,i) = 0;
            else
                rate(u,i) = rate(u,i) / normRate(u,i);
            end
        end
    end
    % To prevent rate of zero that messes calculations
    rate(rate == 0) = 0.00001;

end
% for compatibility with other functions
alpha = alphas;
beta = betas;
gamma = gammas;

% figure;
% plot(logl)
save resses 'Aij' 'rate' 'logl' 'beta' 'alpha' 'gamma'
cd ..
% signal end of running of this task
save(['taste' num2str(taste) 'rep' sindex 'finished'], 'index')
