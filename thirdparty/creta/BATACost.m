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

function [cost] = BATACost(RT,edges,T)
% This function determines BATA cost for a given solution T
% Inputs:
% RT: 3XN matrix of (Cj-Ci)/norm(Cj-Ci)
% edges: MX2 matrix of edge
% T: 3XN matrix of camera locations
% Outputs:
% cost: Mean error of all the edge costs

    T_Diff=T(:,edges(:,2))-T(:,edges(:,1));    
    T_DiffNorm=sum(T_Diff.*RT)./sum(T_Diff.*T_Diff);
    T_DiffNorm(T_DiffNorm<0)=0;
    costall=vecnorm(T_Diff.*T_DiffNorm-RT,2,1);
    costall(isnan(costall))=[];
    cost=mean(costall,2);
end