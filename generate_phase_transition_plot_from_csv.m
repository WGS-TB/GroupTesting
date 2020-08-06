format shortG
gcf;

set(gcf,'Color',[1 1 1]);

%Position plot at left hand corner with width 10 and height 10.
% set(gcf, 'PaperPosition', [0 0 20 10]); 
set(gcf, 'PaperPosition', [0 0 20 10]); 
%Set the paper to have width 10 and height 10.
% set(gcf, 'PaperSize', [20 10]);
set(gcf, 'PaperSize', [20 10]);
set(gcf,'Position',[200 200 1600 1600])

% open ~/Downloads/CM.csv
% CM = CM{:,:};

svals = CM(:,4);
mvals = CM(:,3);
TP = CM(:,12);
TN = CM(:,9);
FP = CM(:,10);
FN = CM(:,11);

svals = reshape(svals,[20,10,5]);
mvals = reshape(mvals,[20,10,5]);
TP = reshape(TP,[20,10,5]);
TN = reshape(TN,[20,10,5]);
FP = reshape(FP,[20,10,5]);
FN = reshape(FN,[20,10,5]);

successes = (FN <= ceil(svals/10.0));
errors = (FN + FP)./svals;

for i = 1:5
    svals(:,:,i) = flipud(svals(:,:,i));
    mvals(:,:,i) = flipud(mvals(:,:,i));
    TP(:,:,i) = flipud(TP(:,:,i));
    TN(:,:,i) = flipud(TN(:,:,i));
    FP(:,:,i) = flipud(FP(:,:,i));
    FN(:,:,i) = flipud(FN(:,:,i));
    successes(:,:,i) = flipud(successes(:,:,i));
    errors(:,:,i) = flipud(errors(:,:,i));
end

avg_FN = sum(FN,3)./5.0;
avg_FP = sum(FP,3)./5.0;
avg_TN = sum(TN,3)./5.0;
avg_TP = sum(TP,3)./5.0;
prob_success = sum(successes,3)./5.0;
avg_err = sum(errors,3)./5.0;

subplot(2,2,1)
imshow(imresize(prob_success,32.0,'nearest'),[min(prob_success(:)),max(prob_success(:))])
cbar = colorbar; cbar.Ticks = linspace(min(prob_success(:)),max(prob_success(:)),5);
% cbar.Position = cbar.Position-[-0.03, 0.000, -0.000, 0.02];
title('Prob. success ($\mathrm{FN}\le\lceil s/10 \rceil$)','Interpreter','LaTeX');
xlabel('$\delta=m/N$','Interpreter','LaTeX');
ylabel('$\rho=k/m$','Interpreter','LaTeX');


subplot(2,2,2)
imshow(imresize(avg_err,32.0,'nearest'),[min(avg_err(:)),max(avg_err(:))])
cbar = colorbar; cbar.Ticks = linspace(min(avg_err(:)),max(avg_err(:)),5);
% cbar.Position = cbar.Position-[-0.03, 0.000, -0.000, 0.02];
title('Relative $\ell_1$ error','Interpreter','LaTeX');
xlabel('$\delta=m/N$','Interpreter','LaTeX');
ylabel('$\rho=k/m$','Interpreter','LaTeX');

subplot(2,2,3)
imshow(imresize(avg_FN,32.0,'nearest'),[min(avg_FN(:)),max(avg_FN(:))])
cbar = colorbar; cbar.Ticks = linspace(min(avg_FN(:)),max(avg_FN(:)),5);
% cbar.Position = cbar.Position-[-0.03, 0.000, -0.000, 0.02];
title('False negatives','Interpreter','LaTeX');
xlabel('$\delta=m/N$','Interpreter','LaTeX');
ylabel('$\rho=k/m$','Interpreter','LaTeX');

subplot(2,2,4)
imshow(imresize(avg_FP,32.0,'nearest'),[min(avg_FP(:)),max(avg_FP(:))])
cbar = colorbar; cbar.Ticks = linspace(min(avg_FP(:)),max(avg_FP(:)),5);
% cbar.Position = cbar.Position-[-0.03, 0.000, -0.000, 0.02];
title('False positives','Interpreter','LaTeX');
xlabel('$\delta=m/N$','Interpreter','LaTeX');
ylabel('$\rho=k/m$','Interpreter','LaTeX');

font_size = 26;

set(findall(gcf,'type','text'),'fontSize',font_size);
set(findall(gcf,'type','axes'),'fontsize',font_size-4);
set(gcf,'DefaultTextFontSize',font_size);
