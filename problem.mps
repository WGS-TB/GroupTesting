NAME	Problem
OBJSENSE
	MIN
OBJNAME
	OBJ
ROWS
 N   OBJ
 G   cp0
 G   cp1
 G   cp2
 G   cn3
COLUMNS
    w0	OBJ	1
    w0	cp0	1
    w1	OBJ	1
    w1	cp2	1
    w2	OBJ	1
    w2	cp0	1
    w2	cp1	1
    w2	cp2	1
    w3	OBJ	1
    w3	cp0	1
    w3	cp2	1
    w4	OBJ	1
    w4	cn3	-1
    w5	OBJ	1
    w5	cp2	1
    ep0	OBJ	0.1
    ep0	cp0	1
    ep1	OBJ	0.1
    ep1	cp1	1
    ep2	OBJ	0.1
    ep2	cp2	1
    en3	OBJ	0.2
    en3	cn3	1
RHS
    RHS1	cp0	1
    RHS1	cp1	1
    RHS1	cp2	1
BOUNDS
   BV BND	w0
   BV BND	w1
   BV BND	w2
   BV BND	w3
   BV BND	w4
   BV BND	w5
   LO BND	ep0	0
   UP BND	ep0	1
   LO BND	ep1	0
   UP BND	ep1	1
   LO BND	ep2	0
   UP BND	ep2	1
   BV BND	en3
ENDATA