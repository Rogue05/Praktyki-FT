clear;close;clear classes;
filename = "respotkanieponiedziaek\\28.06.2021Sample208PowerHigh.txt";

mod = py.importlib.import_module('fin');
py.importlib.reload(mod);

py.fin.process_name(filename)

[p, stderr, r, t, y] = process_file(filename,0.019,0.025,py.fin.Exp2_rise);

subplot(211);
plot(t,y);hold on;
plot(t,eval_model(py.fin.Exp2_rise,t-0.019,p));
subplot(212);plot(t,r);

