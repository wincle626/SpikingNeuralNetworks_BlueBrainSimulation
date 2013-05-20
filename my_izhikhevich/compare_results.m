subplot(2,1,1)
izhik_orig
subplot(2,1,2)
while ( true ) 
    load firings.dat 
    plot(firings(:,1), firings(:,2), 'r.')
    disp('press RETURN to load and plot next run when ready');
    pause
end
