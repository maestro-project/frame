import pandas as pd

def change_F_to_S(input_file, output_file='./test.csv'):
    ''' Converts Csv used in Frame to scale-sim pattern for Convolutions'''
    operator_list = pd.read_csv(input_file).values.tolist()  
    out_operator_list = []
    out_batch_list = []
    for i,in_op in enumerate(operator_list):
        Batch = 1
        if in_op[-1] == 1:  # Conv
            out_op = ["conv-"+str(i), in_op[2], in_op[3], in_op[4], in_op[5], in_op[1], in_op[0], 1]
        elif in_op[-1] == 3:    # GEMM
            out_op = ["GEMM-"+str(i),in_op[0], in_op[2], 1, in_op[2], 1, in_op[1], 1] 
        elif in_op[-1] == 4:
            Batch =   in_op[0]
            out_op = ["Logit-"+str(i),in_op[1], in_op[3], 1, in_op[3], 1, in_op[2], 1]
        elif in_op[-1] == 5:
            Batch =  in_op[0]
            out_op = ["Attend-"+str(i),in_op[1], in_op[2], 1, in_op[2], 1, in_op[3], 1]
        out_operator_list.append(out_op)
        out_batch_list.append(Batch)

    columns = ['Layer Name', 'IFMAP height', 'IFMAP width', 'Filter height', 'Filter width', 'Channels', 'Num filters', 'Stride']
    out_df = pd.DataFrame(out_operator_list, columns=columns)
    out_df[' '] = pd.Series()

    out_df.to_csv(output_file, index=False)
    return out_df, out_batch_list

def conv_change_F_to_S(input_file, output_file='./test.csv'):
    ''' Converts Csv used in Frame to scale-sim pattern for Convolutions'''
    df = pd.read_csv(input_file)  
    ## Frame assumes  K, C, Y, X, R, S
    ##    input_a = (B, C, Y, X)
    ##    input_w = (K, C, R, S)
    ##    output = (B, K, Y, X)
    ##        Frame annotation : Scale-sim annotation
    ##    Input Batch N        :
    ##    Output Channel K     :   Num filters
    ##    Input Channel C      : Channels
    ##    Filter Row R         : Filter height
    ##    Filter Column S      : Filter width
    ##    Input Row Y          : IFMAP height
    ##    Input Column X       : IFMAP width
    ##    Output Row Y’
    ##    Output Column X’

    ## Scale sim expects "Layer name","IFMAP height","IFMAP width","Filter height","Filter width","Channels","Num filter","Stride height","Stride width"
    out_df = pd.DataFrame()
    out_df['Layer Name'] = range(len(df))
    out_df['IFMAP height'] = df['Y']
    out_df['IFMAP width'] = df['X']
    out_df['Filter height'] = df['R']
    out_df['Filter width'] = df['S']
    out_df['Channels'] = df['C']
    out_df['Num filters'] = df['K']
    out_df['Strides'] = 1
    out_df[' '] = None
    
    # df = df.drop(['K','C','Y','X','R','S','T'], axis=1)

    out_df.to_csv(output_file, index=False)
    return out_df

def gemm_change_F_to_S(input_file, output_file='./test.csv'):
    ''' Converts Csv used in Frame to scale-sim pattern for GEMM'''
    df = pd.read_csv(input_file)  
    ## Frame assumes  B, M, N, K = self.dim[:self.get_effective_dim_len()]
    #   input_a = (B, K, N)
    #   input_w = (M, K)
    #   output = (B, M, N)

    ##        Frame annotation : Scale-sim annotation
    ##    MK * BKN = BMN    :   MK * KN = MN
    ## Scale-sim doesn't have batching support.


    # Entries: layer name, Ifmap h, ifmap w, filter h, filter w, num_ch, num_filt, stride h, stride w
    # entries = [layer_name, m, k, 1, k, 1, n, 1, 1]
    
    ## TODO: Deal with logit and attend layers.
    out_df = pd.DataFrame()
    out_df['Layer Name'] = range(len(df))
    out_df['IFMAP height'] = df['M']
    out_df['IFMAP width'] = df['K']
    out_df['Filter height'] = 1
    out_df['Filter wdith'] = df['K']
    out_df['Channels'] = 1
    out_df['Num filters'] = df['N']
    out_df['Strides'] = 1
    out_df[' '] = None
    
    # df = df.drop(['M','N','K','X','R','S','T'], axis=1)

    out_df.to_csv(output_file, index=False)
    return out_df