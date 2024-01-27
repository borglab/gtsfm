% Copyright (C) 2022 Computer Vision Lab, Electrical Engineering, 
% Indian Institute of Science, Bengaluru, India.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above
%       copyright notice, this list of conditions and the following
%       disclaimer in the documentation and/or other materials provided
%       with the distribution.
%     * Neither the name of Indian Institute of Science nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
% DEALINGS IN THE SOFTWARE.
% 
% Author: Lalit Manam
% 
% This file is a part of the implementation for the paper:
% Lalit Manam and Venu Madhav Govindu, Correspondence Reweighted 
% Translation Averaging, Proceedings of the European Conference on 
% Computer Vision 2022

% Demo for CReTA framework
clear; close all
% parpool;

extractParallelRigidSubgraph=true;
extractParallelRigidSubgraphIter=false;
loadGTComponent=false;
loadRavgFilteredGraph=true;

computeInitialRelTransIRLS=false;
computeInitialRelTransRANSAC=true;

stepNormConvTol=1e-6; 
TACostRelConvTol=1e-5;

TAMethod='RLUD'

% Translation Averaging Parameters
if(strcmp(TAMethod,'RLUD'))
    maxIters=5000;
    % RLUD Parameters
    TAparams.delta = 10^-5;
    TAparams.numofiterinit = 50;
    TAparams.numofouteriter = 20;    
    TAparams.robustthre = 10^-1;
elseif(strcmp(TAMethod,'BATA'))
    maxIters=500;
    % Parameters for RLUD-BATA
    TAparams.delta = 10^-5;
    TAparams.numofiterinit = 50;
    TAparams.numofouteriter = 20;
    TAparams.numofinneriter = 5;
    TAparams.robustthre = 10^-1;
else
    error('Invalid TA method');
end

% Relative translation estimation parameters
RTparams.SIGMA=1e-2;

% Filter Edges in Iteration
FEparams.maxAngleDeg=5;

%% Load data
%load('dummy_data.mat');
matches_hash = "6bce599e687072b0c2a6421e66c30eb6fee27c8e";
load("/home/tdriver6/Documents/gtsfm/ta_input_" + matches_hash + ".mat");
matches = reshape(matches, [length(matches) 1]);
edges = edges + 1;
maxImages = max(edges(:));
% The data should contain the following variables:
% Graph has N nodes and M edges.
% RT: 3XM matrix of relative translation directions (Tij=Rj*(Ti-Tj)).
% edges: Mx2 matrix of camera pairs.RT
% matches: Mx1 cell each containing Kx4 matrix.
%   Each matrix contains the point correspondences for each edge.
%   Each row in the matrix contains (xi,yi,xj,yj) where xi,yi and xj,yj
%   are coordinates for images i,j respectively afer correcting for camera intrinsics.
% R_avg: 3X3XN matrix of absolute rotations.
% maxImages: Maimum No. of images in the dataset (used for indexing)
% NOTE: Ensure that the graph is connected and has the maximal parallel
% rigid component.

%% Compute unit norm feature vectors
for i=1:size(matches,1)
    tempMat=matches{i}; 
    numInliers=size(tempMat,1);
    x1=[tempMat(:,1:2),ones(numInliers,1)];
    x1=x1./vecnorm(x1,2,2);
    x2=[tempMat(:,3:4),ones(numInliers,1)];
    x2=x2./vecnorm(x2,2,2);
    matches{i}=[x1,x2];
end

%% Pre processing matches
for i=1:size(matches,1)
    tempMat=matches{i};
    tempMat(:,1:3)=tempMat(:,1:3)*R_avg(:,:,edges(i,1));
    tempMat(:,4:6)=tempMat(:,4:6)*R_avg(:,:,edges(i,2));
    matches{i}=cross(tempMat(:,4:6),tempMat(:,1:3));
end

%% Pre processing relative translations
for i=1:size(edges,1)
    RT(:,i)=-R_avg(:,:,edges(i,2))'*RT(:,i);
end

%% CReTA
iter=1; stepNorm=1; tacostprev=100; tacost=1;
T_avg=[];

while(iter<=maxIters && stepNorm>stepNormConvTol && abs(tacostprev-tacost)/tacostprev>TACostRelConvTol)
    % Run Translation Averaging
    C_avg_prev=T_avg;
    tacostprev=tacost;
    [T_avg,tacost,ed_ret_idx] = RunTA(edges,RT,T_avg,maxImages,TAparams,TAMethod);
    
    % Remove edges that were filtered out
    eidx=~ed_ret_idx;
    edges(eidx,:)=[];
    RT(:,eidx)=[];
    matches(eidx)=[];    
    
    % Filter Edges in iteration
    RT_camSol=T_avg(:,edges(:,2))-T_avg(:,edges(:,1));
    RT_camSol=RT_camSol./vecnorm(RT_camSol,2,1);
    dotProduct=sum(RT.*RT_camSol,1);
    angles = abs(acos(dotProduct));
    SolCalcDiffAngles = angles*180/pi;
    
    eidx=SolCalcDiffAngles>FEparams.maxAngleDeg;
    edges(eidx,:)=[];
    RT(:,eidx)=[];
    matches(eidx)=[];

    % Extract largest connected component
    G=graph(edges(:,1),edges(:,2));
    bins = conncomp(G,'OutputForm','vector');
    nodes = find(bins==mode(bins)); % Map from new (idx) to old (value)

    eidxRet=ismember(edges(:,1),nodes)&ismember(edges(:,2),nodes);
    eidx=~eidxRet;
    edges(eidx,:)=[];
    RT(:,eidx)=[];
    matches(eidx)=[];
    
    % Compute relative translations    
    for i=1:size(edges,1)
        tij=(T_avg(:,edges(i,2))-T_avg(:,edges(i,1)));
        tij=tij/norm(tij);        
        tempMat=matches{i};
        err_vec=abs(tempMat*tij);        
        % Remove points at 1st iteration
        if(iter==1)
            s_err=sort(err_vec);
            remPtsIdx=err_vec>s_err(ceil(numel(s_err)*0.75));
            err_vec(remPtsIdx)=[];
            tempMat(remPtsIdx,:)=[];
            matches{i}=tempMat;
        end        
        w_vec=1./(1+(err_vec/RTparams.SIGMA).^2);
        [~,s,v]=svd(tempMat'*(w_vec.*tempMat));        
        tij_ref=v(:,end);
        if(tij'*tij_ref<0)
            tij_ref=-tij_ref;
        end
        RT(:,i)=tij_ref;  
    end
    
    % Update parameters
    if(iter>1)
        stepDiff=C_avg_prev-T_avg;
        stepDiff(:,any(isnan(stepDiff),1))=[];
        stepNorm=mean(vecnorm(stepDiff,2,1));        
    end
    disp(['Iteration: ',num2str(iter),' done']);
    iter=iter+1;
end

disp('Completed!');

save("/home/tdriver6/Documents/gtsfm/ta_output_" + matches_hash + ".mat", 'T_avg', 'edges')

plot3(T_avg(1,:),T_avg(2,:),T_avg(3,:), ".");
axis equal;
