procs1 = [4, 16, 64];
procs2 = [1, 4, 16, 64];

procs1_run2 = [4, 16, 64];
procs2_run2 = [1, 4, 16, 64];

time1 = [0.198677, 0.566894, 1.327216];           % weak scaling study
time2 = [2.272998, 0.641283, 0.547023, 0.889463];           % strong scaling study
speedup = max(time2) * time2.^-1;
idealspeedup = procs2 / min(procs2);


time1_run2 = [0.198662, 0.528984, 1.062454];
time2_run2 = [2.273759, 0.643935, 0.560727, 0.940972];
speedup_run2 = max(time2_run2) * time2_run2.^-1;
idealspeedup_run2 = procs2_run2 / min(procs2_run2);

figure(1)

str_title = sprintf('Weak Scaling study of 2D Jacobian Smoother with MPI');
title(str_title);
plotspec = 'b-';
plot(procs1, time1, plotspec);
hold on;
plotspec = 'r-';
plot(procs1_run2, time1_run2, plotspec);
xlabel('No of processes')
ylabel('Time')
legend('Run 1', 'Run 2')

figure(2)

str_title = sprintf('Strong Scaling study of 2D Jacobian Smoother with MPI');
title(str_title);
plotspec = 'b-';
plotspec2 = 'b--';
plot(procs2, speedup, plotspec, procs2, idealspeedup, plotspec2);
hold on;
plotspec3 = 'r-';
plotspec4 = 'r--';
plot(procs2_run2, speedup_run2, plotspec3);
xlabel('No of processes')
ylabel('Speedup')
legend('Speedup Run 1', 'Ideal Speedup Run', 'Speedup Run 2')
