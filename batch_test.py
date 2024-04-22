import torch
import torchvision
import torchvision.transforms as transforms

# 데이터 로더 설정
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data/data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/data', train=False, download=True, transform=transform)

# 간단한 CNN 모델
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
)
model.cuda()

batch_sizes = [16, 32, 64, 128]  # 테스트할 배치 크기 목록

for batch_size in batch_sizes:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    data_iter = iter(trainloader)
    images, labels = next(data_iter)
    images = images.cuda()

    # 메모리 사용량 측정 전
    torch.cuda.reset_max_memory_allocated()  # 최대 메모리 할당량 리셋
    model(images)  # 모델 실행

    # 메모리 사용량 확인
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    
    print(f'Batch Size: {batch_size}, Allocated Memory: {allocated_memory}, Free Memory: {free_memory}')
    
    # 실제 모델 훈련 코드는 생략됨
