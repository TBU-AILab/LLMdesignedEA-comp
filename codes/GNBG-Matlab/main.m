%***********************************GNBG***********************************
%Author: Danial Yazdani
%Last Edited: December 14, 2023
%Title: Generalized Numerical Benchmark Generator (GNBG)
% --------
%Description: 
% This MATLAB code implements the Differential Evolution (DE) 
% optimization algorithm (DE/rand/1/bin) to solve a set of 24 problem instances
% generated by the Generalized Numerical Benchmark Generator (GNBG). The GNBG 
% parameter setting for each problem instance is stored in separate '.mat' files
% for ease of access and execution. The code is designed to work directly with 
% these pre-configured instances, allowing for a streamlined and efficient 
% optimization process. Note that the code operates independently of the GNBG
% generator file and only uses the saved configurations of the instances.
% --------
%Refrence: 
%           D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized
%           and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv prepring	arXiv:2312.07083, 2023.
% 
%           AND
% 
%          A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global
%          Optimization," arXiv preprint arXiv:2312.07034, 2023.
%
%If you are using GNBG and this code in your work, you should cite the references provided above.    
% --------
% License:
% This program is to be used under the terms of the GNU General Public License
% (http://www.gnu.org/copyleft/gpl.html).
% Author: Danial Yazdani
% e-mail: danial DOT yazdani AT gmail DOT com
% Copyright notice: (c) 2023 Danial Yazdani
%************************************************************************** 
close all;clear all;clc;%#ok<CLALL> 
RunNumber = 5;
Error = NaN(1,RunNumber);
AcceptancePoints = NaN(1,RunNumber);
ProblemIndex = 20;% Choose a problem instance range from f1 to f24
for RunCounter=1 : RunNumber
    disp(['RunCounter=', num2str(RunCounter)]);
    %% Preparation and loading of the GNBG parameters based on the chosen problem instance
    clear GNBG DE;
    if ProblemIndex >= 1 && ProblemIndex <= 24
        filename = sprintf('f%d.mat', ProblemIndex);
        load(filename, 'GNBG');%Load GNBG parameter values associated with the chosen problem instance.
    else
        error('ProblemIndex must be between 1 and 24.');
    end
    rng('shuffle');%Set a random seed for the optimizer
    %% Optimizer part
    DE.PopulationSize = 100;
    DE.Dimension = GNBG.Dimension;
    DE.LB = GNBG.MinCoordinate;
    DE.UB = GNBG.MaxCoordinate;
    DE.X = DE.LB + (DE.UB - DE.LB)*rand(DE.PopulationSize,DE.Dimension);
    [DE.FitnessValue,GNBG] = fitness(DE.X,GNBG);
    DE.Donor = NaN(DE.PopulationSize,DE.Dimension);
    DE.Cr = 0.9;
    DE.F = 0.5;
    [~,DE.BestID] = min(DE.FitnessValue);
    DE.BestPosition = DE.X(DE.BestID,:);
    DE.BestValue = DE.FitnessValue(DE.BestID);
    %% main loop of the optimizer
    while 1
        [~,DE.BestID] = min(DE.FitnessValue);
        if DE.FitnessValue(DE.BestID)<DE.BestValue
            DE.BestPosition = DE.X(DE.BestID,:);
            DE.BestValue = DE.FitnessValue(DE.BestID);
        end
        %% Mutation
        R = NaN(DE.PopulationSize,3);
        for ii=1 : DE.PopulationSize
            tmp = randperm(DE.PopulationSize);
            tmp(tmp==ii)=[];
            R(ii,:) = tmp(1:3);
        end
        DE.Donor = DE.X(R(:,1),:) + DE.F.*(DE.X(R(:,2),:)-DE.X(R(:,3),:));%DE/rand/1
        %% Crossover==>binomial
        DE.OffspringPosition = DE.X;%U
        K = sub2ind([DE.PopulationSize,DE.Dimension],(1:DE.PopulationSize)',randi(DE.Dimension,[DE.PopulationSize,1]));
        DE.OffspringPosition(K) = DE.Donor(K);
        CrossoverBinomial = rand(DE.PopulationSize,DE.Dimension)<repmat(DE.Cr,DE.PopulationSize,DE.Dimension);
        DE.OffspringPosition(CrossoverBinomial) = DE.Donor(CrossoverBinomial);
        %% boundary checking
        LB_tmp1 = DE.OffspringPosition<DE.LB;
        LB_tmp2 = ((DE.LB + DE.X).*LB_tmp1)/2;
        DE.OffspringPosition(LB_tmp1) = LB_tmp2(LB_tmp1);
        UB_tmp1 = DE.OffspringPosition>DE.UB;
        UB_tmp2 = ((DE.UB + DE.X).*UB_tmp1)/2;
        DE.OffspringPosition(UB_tmp1) = UB_tmp2(UB_tmp1);
        [DE.OffspringFitness, GNBG] = fitness(DE.OffspringPosition(:,1:DE.Dimension), GNBG);
        %% Selection==>greedy
        better = DE.OffspringFitness < DE.FitnessValue;
        DE.X(better,:) = DE.OffspringPosition(better,:);
        DE.FitnessValue(better) = DE.OffspringFitness(better);
        if  GNBG.FE >= GNBG.MaxEvals%When termination criteria has been met
            break;
        end
    end
    %% Storing results of each run
    Error(1,RunCounter) = abs(GNBG.BestFoundResult - GNBG.OptimumValue);
    AcceptancePoints(RunCounter) = GNBG.AcceptanceReachPoint;
end
%% Output
nonInfIndices = isfinite(AcceptancePoints);
nonInfValues = AcceptancePoints(nonInfIndices);
disp(['Average FE to reach acceptance result: ', num2str(mean(nonInfValues)),'(',num2str(std(nonInfValues)),')']);
disp(['Acceptance Ratio: ', num2str((sum(nonInfIndices) / length(AcceptancePoints)) * 100)]);
disp(['Final result: ', num2str(mean(Error(1,:))),'(',num2str(std(Error(1,:))),')']);