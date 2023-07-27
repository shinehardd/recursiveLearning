# 강화학습 프레임워크

##  설치 (Installation)
1. PyTorch 설치
   * 최신 버전으로 설치
   * CPU 버전
```bash
$ pip3 install torch torchvision torchaudio
```
   * GPU 버전은 각자 환경에 맞게 설치 
2. requirement.txt 파일 설치 <br>
```bash
$ pip install -r requirement.txt
```
## 사용법 (Usage)
* main.py 실행  agent, environment 이름 지정
  * --agent: 에이전트 이름 {reinforce, a3c, dqn, ppo}
  * --env: 환경 이름 {:}'CartPole-v1', 'LunarLanderContinuous-v2'}
  
```bash
$ python main.py --agent ppo --env CartPole-v1
$ python main.py -a ppo -e CartPole-v1
```

## 설정 항목 (Configuration) 
#### 공통 부분 (Common Part)
##### GPU
  * **use_cuda**: True
    * GPU가 없을 때 use_cuda를 True로 설정하면 자동으로 False로 만듦
  * **device_num**: 0
    * 사용할 GPU 번호 (CUDA device number)

##### 기타 (Etc)
  * **epsilon**: 0.0000001
    * 작은 상수로 산술 연산을 할 때 수치적 안정성을 위해 사용
    * 예를 들어, 나누기를 할 때 분모가 0이 되지 않게 분모에 더해줌 

##### 옵티마이저 (Optimizer)
  * **optim_betas** : [0.9, 0.999]
    * Adam alpha -> beta1, beta2
  * **optim_eps**: 0.00001
    * Adam에서 사용하는 epsilon 값
  * **torch_deterministic**: True
    * PyTorch 학습 시 random number를 고정했을 때 동일한 학습 결과가 나오게 하는 옵션
    * 단, 학습 성능이 조금 느려질 수 있음

##### 환경 (Environment)
  * **env_wrapper**: 'opengym'
    * 환경을 제공하는 패키지 종류
  * **env_name**: 'CartPole-v1'
    * 환경 이름
    * main.py에서는 argument로 받는 환경 이름을 config에 있는 것보다 우선순위를 높여서 처리
  * **n_envs**: 1
    * 학습 시 생성할 환경 개수 (현재는 여러 환경은 지원하지 않음)

##### 환경 출력 (Rendering)
  * **render**: False
    * 환경을 화면으로 보여줄지 여부
    * 화면을 보여주면 환경 실행 속도가 너무 늦어지므로 학습할 때는 False로 두도록 함

##### 로깅 (Logging options)
  * **log_interval**: 2000
    * 로깅 주기 (Log summary of stats after every {} time steps)
  * **use_tensorboard**: True
    * Tensorboard에 로깅할지 여부

##### 체크포인트 파일 (Checkpoint)
  * **save_model**: True
    * 모델 체크포인트를 저장할지 여부 (Save the models to disk)
  * **save_model_interval**: 20000
    * 체크포인트 저장 주기 (Save models after this many steps)
  * **checkpoint_path**: ""
    * 체크포인트 저장 경로
  * **load_step**: 0
    * 로드할 체크포인트 파일이 있는 스텝 번호
    * Load model trained on this many timesteps (0 if choose max possible)
  * **local_results_path**: "results" 
    * 로그 파일이나 모델 파일을 저장할 디렉토리

#### 학습 관련 부분 (Training Part)
##### 훈련 모드 (Training Mode)
  * **training_mode** : True
    * 훈련 모드에서는 True, 추론 모드에서는 False
  * **trained_model_path**: ""
    * 추론 모드에서 지정된 파일 경로에 있는 모델을 읽어서 추론 
  * **test_mode_max_episodes**: 100
    * 추론 모드에서 실행할 최대 에피소드 수 (maximum number of episodes)

##### 훈련 스텝 (Training Steps)
  * **max_environment_steps**: 100000
    * 훈련 모드에서 실행할 최대 환경 스텝 # maximum environment steps
  * **n_steps**: 1000
    * 학습 데이터를 수집하기 위해 실행할 환경 스텝
  * **n_episodes**: 0
    * 학습 데이터를 수집하기 위해 실행할 에피소드 수
    * n_steps과 설정되어 있으면 n_steps가 우선순위가 높음 
  * **n_epochs**: 1
    * Learner가 학습할 때 Policy 학습 epoch 수 (On Policy에서 사용)
  * **gradient_steps**: 64
    * Learner가 학습할 때 iteration 수 (Off Policy에서 사용)
  * **batch_size**: 32
    * 배치 크기

##### 디스카운트 팩터 (Discount factor)
  * **gamma**: 0.99
    * Return 계산 시 사용하는 할인 계수 (discount factor)

##### 학습률 (Learning Rate)
  * **lr_policy**: 0.005
    * Policy 네트워크 학습률
  * **lr_ciritic**: 0.005
    * Value 네트워크 학습률

##### 학습률 스케쥴링 (Learning Rate Annealing)
  * **lr_annealing**: True
    * 학습률 감소를 처리할 지 여부

##### 리플레이 버퍼 워밍업 (Warming Up step)
  * **warmup_step**: 0
    * Replay Buffer가 비어있을 때 Policy를 학습하지 않고 Buffer를 채우는 스텝 수
    * iteration 기준으로 설정

##### 입실론 그리디 (Epsilon greedy)
  * **epsilon_greedy**: False
    * Deterministic Policy를 사용할 때 탐색(exploration)을 위해 action 선택 시에 추가하는 noise 
  * **epsilon_start**: 0.1
    * epsilon 시작 값 
  * **epsilon_finish**: 0.01
    * epsilon 종료 값 
  * **epsilon_anneal_time**: 70000
    * epsilon 감소를 하는 기간 (timestep 기준) 
 
##### 그레이디언트 클립핑 (Gradient Clipping)
  * **grad_norm_clip**: 0.3
    * Gradient의 크기가 임계치를 넘으면 자름 (L2 norm)

##### 리턴과 이득 (Return and Advantage)
  * **advantage_type**: 'gae'
    * 이득 유형 (gae, n_step, mc) (reinforce, a2c, ppo에서 사용)
    * gae: GAE (Generalized Advantage Estimate)
    * n_step: n 스텝
    * mc: 몬테카를로 리턴
  * **return_standardization**: True
    * n_step이나 mc인 경우 리턴 표준화를 할지 여부
  * **gae_standardization**: False
    * gae에서 표준화를 할지 여부
  * **gae_lambda**: 0.98
    * gae에서 n 스텝 평균을 계산할 때 사용하는 가중치 계수 

##### PPO 클립핑 (PPO Clipping)
  * **ppo_clipping_epsilon**: 0.2 
    * PPO 알고리즘에서 old policy와 new policy의 log likelihood 비율을 제한할 때 사용하는 epsilon 값  
  * **clip_schedule**: True
    * epsilon을 감소시킬지 여부

##### 손실 함수 계수 (loss coefficient)
  * **vloss_coef**: 0.2 
    * value loss 계수 (PPO, A2C에서 사용)  
  * **eloss_coef**: True
    * Entropy bonus 계수 (PPO, A2C에서 사용)  


##### 네트워크 (Network)
  * **actor_hidden_dims**: [64, 64, 64]
    * Policy Network의 계층 별 뉴런 수 
  * **critic_hidden_dims**: [64, 64, 64]
    * Value Network의 계층 별 뉴런 수