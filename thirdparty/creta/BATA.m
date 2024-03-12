% Copyright (C) 2022 Computer Vision Lab, Electrical Engineering, 
% Indian Institute of Science, Bengaluru, India.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
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

% This script is modified from the source:
% https://bbzh.github.io/document/BATA.zip

function [t,ed_ret_idx]=BATA(tij_index,tij_observe,param,t_init_given,idxConst)
% Inputs::
% tij_index: 2 by n matrix specifying the edge (i,j)
% tij_observe: 3 by n matrix specifying tij for each edge (i,j), such that
% tj-ti/norm(tj-ti) = tij
% param: Parameters for the method
% t_init_given: 3 by m matrix of camera translation initialization
% idxConst: Camera index which is chosen as origin
% Outputs::
% t: 3 by m matrix specifying the camera translations
% ed_ret_idx: indexes of the edge retained

if(nargin<4)
    t_init_given=[];
end

    numofcam = max(max(tij_index));
    numofobser = size(tij_observe,2);
    Ri_T = repmat(eye(3),numofobser,1);    
    Rj_T=Ri_T;
    
    index_ti_I = [(1:3*numofobser)' (1:3*numofobser)' (1:3*numofobser)'];   % position of coefficient for Ri
    index_ti_J = (tij_index(1,:)-1)*3+1;
    index_ti_J = [index_ti_J index_ti_J+1 index_ti_J+2];
    index_ti_J = index_ti_J(ceil((1:3*size(index_ti_J,1))/3), :);    
    index_tj_I = [(1:3*numofobser)' (1:3*numofobser)' (1:3*numofobser)'];   % position of coefficient for Rj
    index_tj_J = (tij_index(2,:)-1)*3+1;                 
    index_tj_J = [index_tj_J index_tj_J+1 index_tj_J+2];
    index_tj_J = index_tj_J(ceil((1:3*size(index_tj_J,1))/3), :);
    
    At0_full = sparse(index_ti_I,index_ti_J, Ri_T,3*numofobser,3*numofcam)...
       -sparse(index_tj_I,index_tj_J,Rj_T,3*numofobser,3*numofcam);
    
    Z=zeros(4);
    if(isempty(t_init_given))
        Aeq1 = sparse(reshape(repmat(1:numofobser,3,1),[],1),1:3*numofobser,tij_observe)*At0_full;
        Aeq1 = sum(Aeq1);
        beq1 = numofobser;
        Aeq2 = repmat(eye(3),1,numofcam);
        beq2 = zeros(3,1);
        Aeq = [Aeq1;Aeq2];
        beq = [beq1;beq2];
        
        % Initialization with LUDRevised
        Svec = rand(1,numofobser)+0.5;
        Svec = Svec/sum(Svec)*numofobser;
        S = reshape(repmat(Svec,3,1),[],1);
        W = ones(3*numofobser,1);
        ii = 1;
        
        errPrev=1; errCurr=0;       
        while(ii<=param.numofiterinit && abs(errPrev-errCurr)/errPrev>1e-5)
            A = sparse(1:3*numofobser,1:3*numofobser,W)*At0_full;
            B = W.*S.*tij_observe(:);
            X = [(A'*A) Aeq'; Aeq Z]\[(A'*B); beq];
            t = X(1:3*numofcam);
            Aij = reshape(At0_full*t,3,numofobser);
            tij_T = reshape(tij_observe,3,numofobser);
            Svec = sum(Aij.*tij_T)./sum(tij_T.*tij_T);
            Svec(Svec<0)=0; 
            tmp3 = repmat(Svec,[3,1]);
            S = tmp3(:);               
            errPrev=errCurr;
            tmp= sum(reshape(At0_full*t-S.*tij_observe(:),3,[]).^2);            
            Wvec = (tmp + param.delta).^(-0.5); 
            errCurr=sum(sqrt(tmp));
            W = reshape(repmat(sqrt(Wvec),[3,1]),[],1);
            ii = ii + 1;
        end
    else
        t=t_init_given(:);
    end
    
    % RLUD init
    ii = 1;
    node_ret_idx=true(3,numofcam);
    errPrev=1; errCurr=0;
    beq2 = zeros(3,1);
    I=eye(3);
    while(ii<=param.numofouteriter && (abs(errPrev-errCurr)/errPrev)>1e-5)
                
        Aij = reshape(At0_full*t,3,numofobser);
        tij_T = reshape(tij_observe,3,numofobser);
        Svec = sum(Aij.*tij_T)./sum(tij_T.*tij_T);   
                
        [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec,tij_index,node_ret_idx);
        numEdgesRet=sum(ed_ret_idx(1,:));
        numNodesRet=sum(node_ret_idx(1,:));
        
        t2=t(node_ret_idx(:));
        Svec=Svec(ed_ret_idx(1,:));
        At0_fullr=At0_full(ed_ret_idx(:),node_ret_idx(:));
                      
        tij_T=tij_observe(:,ed_ret_idx(1,:));        
        tmp3 = repmat(Svec,[3,1]);
        S = tmp3(:);
        tmp = reshape(At0_fullr*t2-S.*tij_T(:),3,[]);
        errPrev=errCurr;
        errCurr=sum(vecnorm(tmp,2,1));        
        
        Wvec = (sum(tmp.*tmp) + param.delta).^(-0.5);
        W = reshape(repmat(sqrt(Wvec),[3,1]),[],1);
                
        A = sparse(1:3*numEdgesRet,1:3*numEdgesRet,W)*At0_fullr;
        B = W.*S.*tij_T(:);
        Aeq1 = sparse(reshape(repmat(1:numEdgesRet,3,1),[],1),1:3*numEdgesRet,tij_T)*At0_fullr;        
        Aeq1 = sum(Aeq1); 
        beq1 = numEdgesRet; 
        Aeq2 = repmat(I,1,numNodesRet);    
        X = [(A'*A) Aeq1' Aeq2'; [[Aeq1; Aeq2], Z]]\[(A'*B); beq1; beq2];
        t(node_ret_idx(:))=X(1:3*numNodesRet);
        ii = ii + 1;    
    end    
    
    % BATA       
    t=t-repmat(t(3*idxConst-2:3*idxConst),numofcam,1);
    node_ret_idx_temp=true(3,numofcam); node_ret_idx_temp(:,idxConst)=false;
    t=t(node_ret_idx_temp(:));
    At0 = At0_full(:,node_ret_idx_temp(:));
    errPrev=1; errCurr=0; ii=1;
    while(ii<=param.numofouteriter && (abs(errCurr-errPrev)/errPrev)>1e-5) 
                
        Aij = reshape(At0*t,3,numofobser);
        tij_T_weighted = reshape(tij_observe,3,numofobser);
        Svec = sum(Aij.*Aij)./sum(Aij.*tij_T_weighted);
        tmp3=repmat(Svec,[3,1]); S=tmp3(:);
                
        A = sparse(1:length(S),1:length(S),1./S,length(S),length(S))*At0; 
        B = tij_observe(:);
        tmp = sqrt(sum(reshape((A*t-B).^2,3,[])));
        errPrev=errCurr;
        errCurr=sum(tmp(ed_ret_idx(1,:)));
        Wvec=1./(1+((tmp/param.robustthre).^2)); 
        jj =1;
        while(jj<=param.numofinneriter)
                        
            Aij = reshape(At0*t,3,numofobser);
            tij_T_weighted = reshape(tij_observe,3,numofobser);
            Svec = sum(Aij.*Aij)./sum(Aij.*tij_T_weighted);   
                        
            [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec,tij_index,node_ret_idx);
            node_ret_idx_temp=node_ret_idx; node_ret_idx_temp(:,idxConst)=false;
            Svec=Svec(ed_ret_idx(1,:));
            tmp3 = repmat(Svec,[3,1]);
            S_red = tmp3(:); 
            Wvec2=Wvec(ed_ret_idx(1,:));            
            At0_fullr=At0_full(ed_ret_idx(:),node_ret_idx_temp(:));
            tij_T=tij_observe(:,ed_ret_idx(1,:));
            W = reshape(repmat(sqrt(Wvec2),3,1),[],1); 
                        
            A = sparse(1:length(S_red),1:length(S_red),W./S_red,length(S_red),length(S_red))*At0_fullr; 
            B = W.*tij_T(:);            
            t2=(A'*A)\(A'*B);
            node_ret_idx_temp2=node_ret_idx_temp(:,setdiff(1:end,idxConst)); 
            t(node_ret_idx_temp2(:))=t2;
            jj = jj +1;
        end        
        ii = ii + 1;
    end
    t=reshape(t,3,[]);
    t=[t(:,1:idxConst-1),[0;0;0],t(:,idxConst:end)];
    t(~node_ret_idx(:))=NaN;
    t=t-mean(t,2,'omitnan');
    nodes=find(node_ret_idx(1,:));
    ed_ret_idx=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes);    
end

function [ed_ret_idx,node_ret_idx]=extractLargestConnComp(Svec,tij_index,node_ret_idx)
    ed_ret_idx=true(3,size(tij_index,2));
    ed_ret_idx(:,Svec<0)=false; 
    nodes=find(node_ret_idx(1,:));
    eidx=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes);        
    ed_ret_idx(:,~eidx)=false; 
    edges=tij_index(:,ed_ret_idx(1,:));
    
    G=graph(edges(1,:),edges(2,:));
    bins=conncomp(G,'OutputForm','vector');
    nodes = find(bins==mode(bins));    
    ed_ret_idx_temp=ismember(tij_index(1,:),nodes)&ismember(tij_index(2,:),nodes)&ed_ret_idx(1,:);
    ed_ret_idx=repmat(ed_ret_idx_temp,[3,1]);
    node_ret_idx(:)=false;
    node_ret_idx(:,nodes)=true;
end
