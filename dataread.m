function dataout=dataread(filein,line)
fidin=fopen(filein,'r');
nline=0;
while ~feof(fidin) %?�ж��Ƿ�Ϊ�ļ�ĩβ?
    tline=fgetl(fidin); %?���ļ�����?
    nline=nline+1;
    if nline==line
        data=tline;
        dataout = regexp(data, '\t', 'split');
    end
end
fclose(fidin);