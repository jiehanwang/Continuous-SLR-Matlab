function dataout=ChineseDataread(filein)
fidin=fopen(filein,'r');
nline=0;
while ~feof(fidin) %?�ж��Ƿ�Ϊ�ļ�ĩβ?
    tline=fgetl(fidin); %?���ļ�����?
    nline=nline+1;
    dataout{nline} = regexp(tline, ' ', 'split');
%     if nline==line
%         data=tline;
%         dataout = regexp(data, ' ', 'split');
%     end
end
fclose(fidin);