function [y] = eval_model(model,t,p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ret = py.fin.eval_model(py.type(model),py.numpy.array(t),p);
y = cell2mat(cell(py.list(ret)));
end

