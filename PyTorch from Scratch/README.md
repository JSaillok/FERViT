Completed the implementation with pytorch. There were nevertheless some concessions.  
In all dataloaders the last vector is dropped so that the batches created are the same.  
95% of this is caused by wrong architecture (wrong transform or matrix multiplication).
