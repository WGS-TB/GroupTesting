* Problem:    GroupTesting
* Class:      MIP
* Rows:       3
* Columns:    6 (3 integer, 3 binary)
* Non-zeros:  7
* Format:     Fixed MPS
*
NAME          GroupTes
ROWS
 N  R0000000
 G  R1
 G  R2
 G  R3
COLUMNS
    M0000001  'MARKER'                 'INTORG'
    w[0]      R0000000             1   R1                   1
    w[1]      R0000000             1   R3                   1
    w[2]      R0000000             1   R1                   1
    w[2]      R2                   1
    M0000002  'MARKER'                 'INTEND'
    ep[0]     R0000000            10   R1                   1
    ep[1]     R0000000            10   R2                   1
    ep[2]     R0000000            10   R3                   1
RHS
    RHS1      R1                   1   R2                   1
    RHS1      R3                   1
BOUNDS
 UP BND1      w[0]                 1
 UP BND1      w[1]                 1
 UP BND1      w[2]                 1
 UP BND1      ep[0]                1
 UP BND1      ep[1]                1
 UP BND1      ep[2]                1
ENDATA
