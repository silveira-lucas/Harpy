import numpy as np

#%%

def latex_table(filename, M, H, V, corner):
    
    n_lin = M.shape[0]
    n_col = M.shape[1]
    
    with open(filename, 'wt') as f:
#        print('\\begin{table}[H]', f)
#        print('\\centering', f)
        print('\\begin{tabular}', file=f)
        print('{'+(n_col)*'c|'+'c'+'}', file=f)
        print('\\hline', file=f)
        print('\\hline', file=f)
        
        print(corner+' & ', end='', file=f)
        for i_c in range(n_col-1):
            print(H[i_c]+' & ', end='', file=f)
        else:
            i_c +=1
            print(H[i_c]+' \\\\ ', file=f)
            print('\\hline', file=f)
        
        for i_l in range(n_lin):
            print(V[i_l]+' & ', end='', file=f)
            for i_c in range(n_col-1):
                print(M[i_l, i_c]+' & ', end='', file=f)
            else:
                i_c += 1
                print(M[i_l, i_c]+' \\\\ ', file=f)
        
        print('\\hline', file=f)
        print('\\hline', file=f)
        
        print('\\end{tabular}', file=f)

def num2str(num, fmt):
    string = str(fmt %(num))
    return string