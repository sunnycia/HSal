% clc
% clear;
% delete(gcp);

% matlabpool 4

metricsFolder = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/code4metric';
addpath(genpath(metricsFolder))

% save_base = '/data/sunnycia/hdr_works/source_code/hdr_saliency/metric_matlab';
% if ~isdir(save_base)
%     mkdir(save_base);
% end

if ~exist('dsname', 'var')
    printf('dsname variable not exists')
end
if ~exist('sal_dir', 'var')
    printf('sal_dir variable not exists')
end
if ~exist('dens_dir', 'var')
    printf('dens_dir variable not exists')
end
if ~exist('fixa_dir', 'var')
    printf('fixa_dir variable not exists')
end
if ~exist('save_base', 'var')
    printf('save_base variable not exists')
end


cc_msk  = 1;
sim_msk = 1;
jud_msk = 1;
bor_msk = 0;
kl_msk  = 1;
nss_msk = 1;

[filepath, model_name, ext] = fileparts(sal_dir);

% sal_dir = fullfile(sal_base, model_name);

ext = {'*.jpg','*.bmp', '*.png', '*.jpeg', '*.tif'};
saliencymap_path_list = [];
densitymap_path_list = [];
fixationmap_path_list = [];
for e = 1:length(ext)
    saliencymap_path_list = [saliencymap_path_list dir(fullfile(sal_dir, ext{e}))];
    densitymap_path_list = [densitymap_path_list dir(fullfile(dens_dir, ext{e}))];
    fixationmap_path_list = [fixationmap_path_list dir(fullfile(fixa_dir, ext{e}))];
end

saliencymap_path_list = natsortfiles({saliencymap_path_list.name})
densitymap_path_list = natsortfiles({densitymap_path_list.name});
fixationmap_path_list = natsortfiles({fixationmap_path_list.name});

LengthFiles = length(fixationmap_path_list);
saliency_score_CC = zeros(1,LengthFiles);
saliency_score_SIM = zeros(1,LengthFiles);
saliency_score_JUD = zeros(1,LengthFiles);
saliency_score_BOR = zeros(1,LengthFiles);
saliency_score_KL = zeros(1,LengthFiles);
saliency_score_NSS = zeros(1,LengthFiles);

%% CALCULATE METRICS %%
disp('calculate the metrics...');
t1=clock;
for j = 1 : LengthFiles
    smap_path = char(fullfile(sal_dir,saliencymap_path_list(j)));
    density_path = char(fullfile(dens_dir,densitymap_path_list(j)));
    fixation_path = char(fullfile(fixa_dir, fixationmap_path_list(j)));
    
    fprintf('Handling %s', smap_path);

    image_saliency = imread(smap_path);
    image_density = imread(density_path);
    image_fixation = imread(fixation_path);
    % other_map=zeros(1080, 1920);
    [row,col] = size(image_fixation)
    % imresize(image_saliency, size(image_density));

    if cc_msk
        %% CC %%
        saliency_score_CC(j) = CC(image_saliency, image_density);
        fprintf('cc value %s\n', saliency_score_CC(j));

    end
    
    if sim_msk
        %% SIM %% 
        saliency_score_SIM(j) = similarity(image_saliency, image_density);
        fprintf('sim value %s\n', saliency_score_SIM(j));
    end
    
    if jud_msk
        %% AUCJUDD %%
        saliency_score_JUD(j) = AUC_Judd(image_saliency, image_fixation);
    end
    
    if bor_msk
        %% AUCBorji %%
        saliency_score_BOR(j) = AUC_Borji(image_saliency, image_fixation);
    end
    
    if kl_msk
        %% KL %%
        saliency_score_KL(j) = KLdiv(image_saliency, image_density);
    end
        
    if nss_msk
        saliency_score_NSS(j)=NSS(image_saliency, image_fixation);
    end
    
    [saliency_score_CC(j);saliency_score_SIM(j);
                saliency_score_JUD(j);saliency_score_BOR(j);
                saliency_score_KL(j);saliency_score_NSS(j);]
end

saliency_score=[saliency_score_CC;saliency_score_SIM;
                saliency_score_JUD;saliency_score_BOR;
                saliency_score_KL;saliency_score_NSS;]
t2=clock;
time_cost=etime(t2,t1);

save_path = fullfile(save_base, strcat(dsname,'_', model_name,'.mat'))
save(save_path, 'saliency_score','time_cost');
fprintf('%s saved\n',save_path);
