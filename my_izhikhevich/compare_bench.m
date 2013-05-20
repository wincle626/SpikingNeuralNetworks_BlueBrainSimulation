load bench-omp.txt
load bench-nonomp.txt
figure
hold on
scatter(bench_omp(:,1), bench_omp(:,2), 'r')
scatter(bench_nonomp(:,1), bench_nonomp(:,2), 'b')
%subplot(1,2,1)
%boxplot(bench_nonomp(:,2), bench_nonomp(:,1))
%subplot(1,2,2)
%boxplot(bench_omp(:,2), bench_omp(:,1))
legend('with OMP', 'without OMP')
xlabel('number of neurones')
ylabel('simulation time')
