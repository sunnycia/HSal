
hdr_dir='J:\FTP\codel\HDR\HDR-Project-master\GH\HDR_Toolbox-master\HDR'
output_dir='./HDREYE_falsemap'
name_list = dir(hdr_dir);
for i=3:length(name_list)
    hdr_path=fullfile(hdr_dir,name_list(i).name);
    [path,name,ext]=fileparts(hdr_path);
    save_path=fullfile(output_dir,strcat(name,'.jpg'));
    if exist(save_path,'file')
        continue
    end
    hdr = hdrread(hdr_path);
    false_color = FalseColor(hdr, 'log',0);
    
    
    imwrite(false_color,save_path);
end