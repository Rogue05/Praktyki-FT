function [p, stderr, r, t, y] = process_file(filename,t_start,t_end,model)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

ret = py.fin.process_file(filename,t_start,t_end,py.type(model));
p = cell2mat(cell(py.list(ret{1})));
stderr = cell2mat(cell(py.list(ret{2})));
r = cell2mat(cell(py.list(ret{3})));
t = cell2mat(cell(py.list(ret{4})));
y = cell2mat(cell(py.list(ret{5})));
end

