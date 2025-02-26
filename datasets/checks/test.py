def test_dataset(dataset):
    print(f"Testing dataset with {len(dataset)} samples...\n")
    
    for idx in range(len(dataset)):
        try:
            X, edge_index, label, subject = dataset[idx]  # Call __getitem__
            
            # Print shapes and types
            print(f"Index {idx}:")
            print(f"  X shape        : {X.shape} (Expected: [num_timesteps, num_nodes, 3])")
            print(f"  edge_index shape: {edge_index.shape} (Expected: [2, num_edges])")
            print(f"  label type     : {type(label)} (Expected: int or torch.Tensor)")
            print(f"  subject type   : {type(subject)} (Expected: int or string)\n")

            # Optional: Stop after 5 samples to avoid too much output
            # if idx == 4:
            #     print("Stopping early for brevity...")
            #     break
        
        except Exception as e:
            print(f"Error at index {idx}: {e}")