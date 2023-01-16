    # Sample the neighbor number k for synethic nodes    (supported by Pyg）#

    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)  #obtain the degree of each node


    if node_mask is None:
        node_masks = torch.ones_like(degree,dtype=torch.bool)
  
    else:
        node_masks=node_mask.clone()
   
   # obtain the degree distribution. For example, if the number of nodes with degree 1 is 3 and the number of nodes with degree 2 is 5, the degree distribution will be [3,5..] 
    degree_dist = scatter_add(torch.ones_like(degree[node_masks]), degree[node_masks]).to(device).type(torch.float32)
  
   # Sample degree for augmented nodes
   # the parameter [sampling_src_idx] denotes the minor nodes which need to be augmented
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) 

   # prevent the case that the the neighbor numbers of the augmented node from being greater than the origin one.
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])