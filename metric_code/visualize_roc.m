
img_dir = '/data/SaliencyDataset/Image/ETHyma/images'
mat_base='/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/ethyma_reinhard-roc_mat/reinhard'
mat_base_2 = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/ethyma_stackfusion_mink-roc_mat/MINKOWSKI'
ref_dir ='/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/ethyma_stackfusion_mink-roc_mat/MINKOWSKI/MINKOWSKI-fusion-v1_single_mscale_tripleres_resnet50_2018110500:09:34_iter-120000_norm+border+center' 

% img_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
% mat_base = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/hdreye_reinhard-roc_mat/reinhard-toolbox'
% mat_base_2 = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/hdreye_stackfusion_mink-roc_mat/MINKOWSKI'

% ref_dir = '/data/sunnycia/hdr_works/source_code/hdr_saliency/training/metric_code/hdreye_stackfusion_mink-roc_mat/MINKOWSKI/MINKOWSKI-fusion-v1_single_mscale_tripleres_resnet50_2018110500:09:34_iter-120000_norm+border+center';

img_name_list = dir(img_dir);
sub_dir_list = dir(mat_base);
sub_dir_list_2 = dir(mat_base_2);

color_list=['c','g','b','y','m']

for i=3:length(img_name_list)
    img_name = img_name_list(i).name;
    [path,name,ext] = fileparts(img_name);
    mat_name = strcat(name, '.mat')
    save_fig_path = strcat(name,'.png')

    h=figure

    for j=3:length(sub_dir_list)
        cur_dir = fullfile(mat_base,sub_dir_list(j).name)
        mat_path = fullfile(cur_dir, mat_name)
        load(mat_path);

        plot(fp,tp,strcat(':',color_list(j-2)),'LineWidth',2);hold on;
        % legend(sub_dir_list(j).name)
        legendInfo{j-2} = [sub_dir_list(j).name];
    end

    for j=3:length(sub_dir_list_2)
        cur_dir = fullfile(mat_base_2,sub_dir_list_2(j).name)
        mat_path = fullfile(cur_dir, mat_name)
        load(mat_path);

        plot(fp,tp,strcat('-',color_list(j-2)),'LineWidth',2);hold on;
        % legend(sub_dir_list(j).name)
        legendInfo{j-4+length(sub_dir_list)} = [sub_dir_list_2(j).name];
    end

    % load(fullfile(ref_dir, mat_name))
    % plot(fp,tp,'LineWidth',6);hold on;
    % legendInfo=[legendInfo 'TSRSN-MSK']
    grid on

    title(sprintf('Sample: %s',name),'fontsize',16);
    xlabel('FP rate','fontsize',16); ylabel('TP rate','fontsize',16);

    legend(legendInfo,'Location','southeast','FontSize',12)
    saveas(gcf,save_fig_path,'png')

    close(h)
end