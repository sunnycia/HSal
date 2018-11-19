met_dir = '/data/sunnycia/hdr_works/source_code/hdr_saliency/matlab-metric/salicon_val'

met_result_list = dir(met_dir);
met_result_list = natsortfiles({met_result_list.name})

for m=1:7


    for i=3:length(met_result_list)
        met_result_path = char(fullfile(met_dir, met_result_list(i)));
        load(met_result_path)
        metric(i-2)=mean(saliency_score(m,:))
    end
    plot(metric);hold on
end