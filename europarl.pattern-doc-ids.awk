BEGIN {
	FS = "\t";
	docid = "00000000";
}

$3 ~ /^((Madam|Mr) President|Resumption of the Session|in writing \.|(- )?\( [A-Z][A-Z] \) )/ {
	docid = sprintf("%08d", FNR);
}

{
	print docid;
}
