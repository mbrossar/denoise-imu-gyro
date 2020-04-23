import torch
from src.lie_algebra import SO3

# define two rotation from roll pitch yaw
rpys = torch.zeros(2, 3).cuda()
rpys[0, 0] = 0.3
rpys[0, 1] = -0.2
rpys[0, 2] = 0.6

# move to matrix
Rots = SO3.from_rpy(rpys[:, 0], rpys[:, 1], rpys[:, 2])

# compute error from matrices
dR = Rots[0].t().mm(Rots[1])
xi = SO3.log(dR.unsqueeze(0)).squeeze()
print("error from SO(3)", xi.norm().item())


# compute error from quaternions
qs = SO3.to_quaternion(Rots)
tmp = (qs[0]*qs[1]).sum()
theta = 2*torch.acos(tmp.abs())
print("error from quaternion", theta.item())

print("diff", (xi.norm()-theta).item())
