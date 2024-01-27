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

function [T_avg,tacost,ed_ret_idx] = RunTA(edges,RT,T_avg,maxImages,param,TAMethod)
% Run Translation Averaging (TA)
% Inputs::
% edges: MX2 matrix of camera pairs
% RT: 3XM matrix of relative translations
% T_avg: 3XN matrix of camera locations
% maxImages: Maximum no of images for indexing
% param: Parameters for TA Method
% TAMethod: 'RLUD' or 'BATA'
% Outputs:
% T_avg: 3XN matrix of camera locations
% tacost: Valur of TA cost at the solution
% ed_ret_idx: Edge indexes used for translation averaging

% Relabelling nodes
nodes=unique(edges(:));
G=graph(edges(:,1),edges(:,2),1:size(edges,1));
D=degree(G); [~,loc]=max(D); 
idxConst=find(nodes==loc);
G1=subgraph(G,nodes);
P=G1.Edges;
edges_sg=P.EndNodes;
idx=find(edges(:,1)>edges(:,2));
edge_idx_map=P.Weight;
RT(:,idx)=-RT(:,idx);
RT_sg=RT(:,edge_idx_map);

if(isempty(T_avg))
    C_avg_temp=[];
else
    C_avg_temp=T_avg(:,nodes);
end

% Translation Averaging
if(strcmp(TAMethod,'RLUD'))
    [C_avg_temp,ed_ret_idx]=RLUD(edges_sg',-RT_sg,param,C_avg_temp);
    tacost=LUDCost(RT_sg,edges_sg,C_avg_temp);
elseif(strcmp(TAMethod,'BATA'))
    [C_avg_temp,ed_ret_idx]=BATA(edges_sg',-RT_sg,param,C_avg_temp,idxConst);
    tacost=BATACost(RT_sg,edges_sg,C_avg_temp);
else
    error('Invalid TA method');
end

% Converting back to original labels
T_avg=nan(3,maxImages);
T_avg(:,nodes)=C_avg_temp;
ed_ret_idx(edge_idx_map)=ed_ret_idx;

end
