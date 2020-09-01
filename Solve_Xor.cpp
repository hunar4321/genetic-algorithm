//Hunar @ Brainxyz.com

#pragma once
#include <iostream>
#include <vector>
#include <math.h>

//////// helper functions;

double gen_weight() {
	return ((double)rand() / (double)RAND_MAX) - 0.5;
}

double activationF(double val) {
	return std::tanh(val);
}

//// class declarations
struct Edge;
struct Node;
struct Layer;
struct Network;

////// class function declarations
struct Edge {
	double fW = gen_weight();
	double cW = fW;
	Node* from = nullptr;
	Node* to = nullptr;

	void mutateEdge(double lr);
	void childBecomesFather();
};

struct Node {
	double fY = 1;
	double cY = 1;
	std::vector<Edge*> from_edges;
	std::vector<Edge*> to_edges;
	void spreadOut();
	void mutateChildern(double lr);
	void connectTo(Network* net, Node* targNode, double weight);
};

struct Layer {
	std::vector<Node*> nodes;
	void appendNodes(long num);
	void resetLayer();
	void spreadOut(double lr, Network* net);
	void activate();
	void fconnectTo(Network* net, Layer* targLayer);
	void sconnectTo(Network* net, Layer* targLayer, long sparse_rate);
};

struct Network {
	std::vector<Edge*> edgeList;
	std::vector<Edge*> activeEdges;
	Layer sensor = Layer();
	Layer hidden = Layer();
	Layer out = Layer();
	Layer bias1 = Layer();

	Network(long, long, long);
	void fullyConnect();
	void sparseConnect(std::vector<double>);
	void setInput(std::vector<double> inputs);
	void forward(std::vector<double> inputs, double lr);
	void updateWeights();
	void mutateWeights(double lr);
	void resetNet();
};


//// implementations //////
void Edge::mutateEdge(double lr) {
	cW = fW + (gen_weight() * lr);
}
void Edge::childBecomesFather() { fW = cW; }

void Node::spreadOut() {
	for (auto e : to_edges) {
		e->to->fY += fY * e->fW;
		e->to->cY += cY * e->cW;
	}
}
void Node::mutateChildern(double lr) {
	for (auto e : to_edges) {
		e->mutateEdge(lr);
	}
}
void Node::connectTo(Network* net, Node* targNode, double weight) {
	auto edge = new Edge();
	edge->fW = weight;
	edge->cW = weight;
	edge->from = this;
	edge->to = targNode;

	net->edgeList.push_back(edge);
	to_edges.push_back(edge);
	targNode->from_edges.push_back(edge);
}

void Layer::appendNodes(long num) {
	for (long i = 0; i < num; i++) {
		nodes.push_back(new Node());
	}

}
void Layer::resetLayer() {
	for (auto n : nodes) {
		n->fY = 0;
		n->cY = 0;
	}
}
void Layer::spreadOut(double lr, Network* net) {
	for (auto n : nodes) {
		if (n->fY > 0) {
			n->spreadOut();
			for (auto e : n->to_edges) {
				net->activeEdges.push_back(e);
			}
		}
	}
}

void Layer::activate()
{
	for (auto n : nodes) {
		n->fY = activationF(n->fY);
		n->cY = activationF(n->cY);
	}
}

void Layer::fconnectTo(Network* net, Layer* targLayer)
{
	for (auto sn : nodes) {
		for (auto tn : targLayer->nodes) {
			sn->connectTo(net, tn, gen_weight());
		}
	}
}

void Layer::sconnectTo(Network* net, Layer* targLayer, long sparse_rate)
{
	for (auto sn : nodes) {
		for (auto tn : targLayer->nodes) {
			if ((gen_weight() + 0.5) < sparse_rate) {
				sn->connectTo(net, tn, gen_weight());
			}
		}
	}
}

Network::Network(long L1, long L2, long L3) {

	sensor.appendNodes(L1);
	hidden.appendNodes(L2);
	out.appendNodes(L3);
	bias1.appendNodes(1);
}

void Network::fullyConnect()
{
	sensor.fconnectTo(this, &hidden);
	hidden.fconnectTo(this, &out);
	bias1.fconnectTo(this, &out);
}

void Network::sparseConnect(std::vector<double> amount)
{
	sensor.sconnectTo(this, &hidden, amount[0]);
	hidden.sconnectTo(this, &out, amount[1]);
	bias1.fconnectTo(this, &out);
}

void Network::setInput(std::vector<double> inputs)
{
	for (long i = 0; i < inputs.size(); i++) {
		sensor.nodes[i]->fY = inputs[i];
		sensor.nodes[i]->cY = inputs[i];
	}
}

void Network::mutateWeights(double lr) {
	for (auto e : edgeList) {
		e->mutateEdge(lr);
	}
}
void Network::forward(std::vector<double> inputs, double lr)
{
	setInput(inputs);
	sensor.spreadOut(lr, this);
	hidden.activate();
	hidden.spreadOut(lr, this);
	bias1.spreadOut(lr, this);
}

void Network::updateWeights()
{
	for (auto e : activeEdges) {
		e->childBecomesFather();
	}
}

void Network::resetNet()
{
	sensor.resetLayer();
	hidden.resetLayer();
	out.resetLayer();
	activeEdges.clear();
}


int main() {

	long epochs = 1000;
	const float lr = 0.1;
	const int ins = 2;
	const int nodes = 10;
	const int out = 1;
	const int nsamples = 4;

	Network net = Network(ins, nodes, out);
	net.fullyConnect();
	//net.sparseConnect({ 0.1, 0.5 });

	// data & labels
	std::vector<std::vector<double>> inputs = { {0, 0} ,{1, 1},{1, 0},{0, 1} };
	std::vector<double> labels= { 0, 0, 1, 1 };

	// training loop
	double fE, cE, label; 
	std::vector<double> inp;
	for (int i = 0; i < epochs; i++) {

		fE = 0; cE = 0; // initialize father Erorr(fE) and child Error(cE)
		for (int j = 0; j < nsamples; j++) {
			inp = inputs[j];
			label = labels[j];
			net.resetNet();
			net.forward(inp, lr);

			// calculates the network error
			fE = fE + std::abs(net.out.nodes[0]->fY - label);
			cE = cE + std::abs(net.out.nodes[0]->cY - label);
		}

		if (fE > cE)
			net.updateWeights();

		net.mutateWeights(lr);

	}

	std::cout << "assess training" << std::endl;

	for (int j = 0; j < labels.size(); j++) {
		net.resetNet();
		inp = inputs[j];
		label = labels[j];
		net.forward(inp, lr);
		std::cout << "target:" << label << " pred:" << net.out.nodes[0]->fY << std::endl;
	}

	std::cin.get();
}
