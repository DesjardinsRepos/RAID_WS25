def decode(s:bytes)->bytes:
	p=[i for i,c in enumerate(s)if 64<c<91 or 96<c<123]
	if len(p)<133:return b""
	l=16+int(''.join('1'if 64<s[i]<91 else'0'for i in p[:5]),2)
	k=l*8;d=p[5:];v=[64<s[i]<91 for i in d];r=[0]*k
	for j in range(k,len(d)):
		y=(0x9E3779B97F4A7C15*j+0xD1B54A32D192ED03)&((1<<64)-1)
		for t in {(j-k)%k,(y&255)%k,((y>>8)&255)%k,((y>>16)&255)%k}:r[t]^=v[j]
	b=''.join('1'if v[i]^r[i]else'0'for i in range(k))
	return bytes(int(b[i:i+8],2)for i in range(0,k,8))