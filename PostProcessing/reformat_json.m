function reformat_json

default_file_tag = 'default_file';
if ispref(mfilename,default_file_tag)
    default_file = getpref(mfilename,default_file_tag);
else
    default_file = pwd;
end
[fileName,pathName,fileIndex] = uigetfile('*.json', 'Select the 1st json file:',default_file);
if fileIndex == 0
    return;
end
json_path = fullfile(pathName,fileName);
setpref(mfilename,default_file_tag,json_path);
[~,baseName] = fileparts(fileName);
output_path = fullfile(pathName,strcat(baseName,'.txt'));
fid_in = fopen(json_path,'r');
json_data = fread(fid_in,Inf,'*char');
fclose(fid_in);

json_data = json_data'; % transpose from column to row
old_json = json_data;

tab_char = char(9);
lf_char = char(10);
cr_char = char(13);
new_json = '';

indent_count = 0;
old_length = numel(json_data);

reformatted_json = replace(old_json,':',char(9));
reformatted_json = replace(reformatted_json,',',[char(13) char(9)]);
reformatted_json = replace(reformatted_json,'{',[char(13) char(9) '>']);
reformatted_json = replace(reformatted_json,'}',[char(13) '<']);

fid_out = fopen(output_path,'w');
fwrite(fid_out,reformatted_json,'char');
fclose(fid_out);




