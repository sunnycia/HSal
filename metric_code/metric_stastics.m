
% base_dir = '/data/sunnycia/hdr_works/source_code/hdr_saliency/matlab-metric/hdreye_hdr';

if ~exist('save_base', 'var')
    printf('save_base variable not exists')
end

mat_list = dir(fullfile(save_base, '*.mat*'));


for j=1:length(mat_list)

    mat_path=fullfile(save_base, mat_list(j).name)
    load(mat_path);
    [met_num,img_num] = size(saliency_score);
    mean_result=zeros(met_num,1);
    std_result = zeros(met_num, 1);
    for k=1:met_num
        metric=saliency_score(k, :);
        [m,n]=find(isnan(metric)==1);
        metric(m, :)=[];
        mean_result(k) = mean(metric);
        std_result(k) = std(metric);
    end
    mean_result'
    std_result'
end
