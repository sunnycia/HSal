 
metricsFolder = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/code4metric';
addpath(genpath(metricsFolder))

% save_base = '/data/sunnycia/hdr_works/source_code/hdr_saliency/metric_matlab';
% if ~isdir(save_base)
%     mkdir(save_base);
% end

npoints = 10; % number of points to sample on ROC curve
colmap = fliplr(colormap(jet(npoints)));
G = linspace(0.5,1,20)';
tpmap = horzcat(zeros(size(G)),G,zeros(size(G))); % green color space
fpmap = horzcat(G,zeros(size(G)),zeros(size(G))); % red color space


if ~exist('sal_dir', 'var')
    printf('sal_dir variable not exists')
end
if ~exist('fixa_dir', 'var')
    printf('fixa_dir variable not exists')
end
if ~exist('save_dir', 'var')
    printf('save_dir variable not exists')
end
[filepath, model_name, ext] = fileparts(sal_dir);

% sal_dir = fullfile(sal_base, model_name);

ext = {'*.jpg','*.bmp', '*.png', '*.jpeg', '*.tif'};
saliencymap_path_list = [];
fixationmap_path_list = [];
for e = 1:length(ext)
    saliencymap_path_list = [saliencymap_path_list dir(fullfile(sal_dir, ext{e}))];
    fixationmap_path_list = [fixationmap_path_list dir(fullfile(fixa_dir, ext{e}))];
end

saliencymap_path_list = natsortfiles({saliencymap_path_list.name})
fixationmap_path_list = natsortfiles({fixationmap_path_list.name});

LengthFiles = length(fixationmap_path_list);
saliency_score_JUD = zeros(1,LengthFiles);

%% CALCULATE METRICS %%
disp('calculate the metrics...');
t1=clock;
for j = 1 : LengthFiles
% parfor j=1:LengthFiles
    smap_path = char(fullfile(sal_dir,saliencymap_path_list(j)));
    fixation_path = char(fullfile(fixa_dir, fixationmap_path_list(j)));
    
    [path,name,ext]=fileparts(smap_path);
    fprintf('Handling %s\n', smap_path);

    image_saliency = imread(smap_path);
    if size(image_saliency,3)==3
        image_saliency = rgb2gray(image_saliency);
    end
    image_saliency=im2double(image_saliency);
    image_fixation = imread(fixation_path);
    % other_map=zeros(1080, 1920);
    [row,col] = size(image_fixation);
    % imresize(image_saliency, size(image_density));

    [score,tp,fp,allthreshes] = AUC_Judd(image_saliency, image_fixation);
    
    N = ceil(length(allthreshes)/npoints);
    allthreshes_samp = allthreshes(1:N:end);

    h=figure; 

    % plot the ROC curve
    tp1 = tp(1:N:end); fp1 = fp(1:N:end);
    plot(fp,tp,'b');hold on;
    for ii = 1:npoints
         plot(fp1(ii),tp1(ii),'.','color',colmap(ii,:),'markersize',20); hold on; axis square;
    end 
    title(sprintf('AUC: %2.2f',score),'fontsize',14);
    xlabel('FP rate','fontsize',14); ylabel('TP rate','fontsize',14);
    save_fig_path=fullfile(save_dir, strcat(name,'.png'));
    saveas(gcf,save_fig_path,'png')

    save_mat_path=fullfile(save_dir, strcat(name,'.mat'));
    save(save_mat_path,'tp','fp','score','allthreshes_samp')

    save_mat_path=fullfile(save_dir, strcat(name,'-tp.csv'));
    csvwrite(save_mat_path,tp)
    save_mat_path=fullfile(save_dir, strcat(name,'-fp.csv'));
    csvwrite(save_mat_path,fp)

    close(h)
    
end
