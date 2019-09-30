import torch

done1 = torch.LongTensor([3, 1, 4])

done2 = done1.argmax(0, keepdim=True)

print(done2)






