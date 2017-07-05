# 초기화 : 입력, 은닉, 출력 노드 수 결정
# 학습 : 학습 데이터들을 통해 학습하고 이에 따라 가중치를 업데이트
# 질의 : 입력을 받아 연산한 후 출력 노드에서 답을 전달

import numpy
import scipy.special

# 신경망 클래스
class neuralNetwork:

    # 신경망 초기화
    def __init__(self, inputnode, hiddennode, outputnode, learningrate):

        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.input_node = inputnode
        self.hidden_node = hiddennode
        self.output_node = outputnode

        self.learning_rate = learningrate

        # numpy.random.rand(a,b) 는 0 에서 1사이의난수 행렬을 만들어준다.
        # 하지만 가중치가 음수인 경우도 있기 때문에 0.5를 빼서 -0.5~0.5 사이의 값을 가지게 한다.
        # self.W_input_hidden = (numpy.random.rand(self.hidden_node, self.input_node) - 0.5)
        # self.W_hidden_output = (numpy.random.rand(self.output_node, self.hidden_node) - 0.5)

        # 가중치는 0을 중심으로 하면 1/루트(들어오는 연결 노드의 개수) 의 표준편차를 가지는 정규분포에 따라 구한다.
        # 첫 번째 파라미터 : 정규분포의 중심, 두 번째 파라미터 : 노드로 들어오는 연결 노드의 개수에 루트를 씌우고 역수를 취한 것을 파이썬 문법으로 나타냄
        # 세 번째 파라미터 : 우리가 원하는 numpy 행렬
        self.W_input_hidden = numpy.random.normal(0.0, pow(self.hidden_node, -0.5), (self.hidden_node, self.input_node))
        self.W_hidden_output = numpy.random.normal(0.0, pow(self.output_node, -0.5), (self.output_node, self.hidden_node))

        # 활성화 함수로 sigmoid 함수 사용
        # lambda 함수는 x를 매개변수로 전달받아 시그모이드 함수인 scipy.special.expit(x)를 반환한다. 활성화 함수를 사용해야한다면 activation_function 호출하면됨
        self.activation_function = lambda x: scipy.special.expit(x)




    # 신경망 학습
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원의 행렬로 변환
        input = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산
        hidden_input = numpy.dot(self.W_input_hidden, input)
        # 은닉 계층에서 나가는 신호를 계산
        hidden_output = self.activation_function(hidden_input)

        # 최종 출력 계층으로 들어오는 신호 계산
        final_input = numpy.dot(self.W_hidden_output, hidden_output)
        # 최종 출력 계층에서 나가는 신호 계산
        final_output = self.activation_function(final_input)

        # 오차 = (실제값 - 계산값)
        output_error = targets - final_output
        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_error = numpy.dot(self.W_hidden_output.T, output_error)

        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.W_hidden_output += self.learning_rate * numpy.dot((output_error * final_output * (1.0 - final_output)), numpy.transpose(hidden_output))
        # 입력 계층과 은닉 계층 간의 가중치 업데이트
        self.W_input_hidden += self.learning_rate * numpy.dot((hidden_error * hidden_output * (1.0 - hidden_output)), numpy.transpose(input))


    # 신경망 테스트
    def ask(self, inputs_list):

        #입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산
        hidden_input = numpy.dot(self.W_input_hidden, inputs)
        # 은닉 계층에서 나가는 신호를 계산
        hidden_output = self.activation_function(hidden_input)

        # 최종 출력 계층으로 들어오는 신호를 계산
        final_input = numpy.dot(self.W_hidden_output, hidden_output)
        # 최정 출력 계층에서 나가는 신호를 계산
        final_output = self.activation_function(final_input)

        return final_output



input_node = 1500
hidden_node = 100
output_node = 3
learningrate = 0.01

test = neuralNetwork(input_node, hidden_node, output_node, learningrate)

trainint_data_file = open("Acc_data.csv", 'r')
trainint_data_list = trainint_data_file.readlines()
trainint_data_file.close()

for record in trainint_data_list:

    all_values = record.split(',')
    input = numpy.asfarray(all_values[0:3])

    target = numpy.zeros(output_node) + 0.01
    # target[int(all_values[0])] = 0.99

    test.train(input, target)