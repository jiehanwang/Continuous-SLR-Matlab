function dataout=ChineseDataread(filein)
fidin=fopen(filein,'r');
nline=0;
while ~feof(fidin) %?判断是否为文件末尾?
    tline=fgetl(fidin); %?从文件读行?
    nline=nline+1;
    dataout{nline} = regexp(tline, ' ', 'split');
%     if nline==line
%         data=tline;
%         dataout = regexp(data, ' ', 'split');
%     end
end
fclose(fidin);