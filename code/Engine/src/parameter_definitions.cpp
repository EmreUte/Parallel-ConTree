/**
 * Implementation derived from Emir Demirovic's "MurTree"
 * For more details, visit: https://bitbucket.org/EmirD/murtree
 */

#include "parameter_handler.h"

ParameterHandler ParameterHandler::DefineParameters() {
	ParameterHandler parameters;

	parameters.DefineNewCategory("Main Parameters");
	parameters.DefineNewCategory("Algorithmic Parameters");

	parameters.DefineStringParameter
	(
		"file",
		"Location to the (training) dataset.",
		"", //default value
		"Main Parameters"
	);

	parameters.DefineIntegerParameter
	(
		"max-depth",
		"Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.",
		3, //default value
		"Main Parameters",
		0, //min value
		20 //max value
	);

    parameters.DefineIntegerParameter
    (
        "max-gap",
        "Maximum difference between the solution of the algorithm and the optimal solution.",
        0, //default value
        "Main Parameters",
        0, //min value
        1000000 //max value
    );

	parameters.DefineFloatParameter
	(
		"max-gap-decay",
		"The decay of the maximum gap.",
		0.0, //default value
		"Main Parameters",
		0.0, //min value
		1.0 //max value
	);

    parameters.DefineIntegerParameter
    (
        "run-number",
        "The number of runs to average over the runtime.",
        1, //default value
        "Main Parameters",
        1, //min value
        10 //max value
    );

	parameters.DefineBooleanParameter
	(
		"print-logs",
		"Determines if the solver should print logging information to the standard output.",
		false,
		"Main Parameters"
	);

	parameters.DefineFloatParameter
	(
		"time",
		"The runtime limit in seconds.",
		600, // default value
		"Main Parameters",
		0, //min value
		INT32_MAX // max value
	);

	parameters.DefineBooleanParameter
	(
		"use-upper-bound",
		"Use upper bounding. ",
		true,
		"Algorithmic Parameters"
	);

	parameters.DefineBooleanParameter
	(
		"enable-gpu",
		"Enable hybrid CPU/GPU execution for deeper subtrees (requires CUDA build).",
		false,
		"Algorithmic Parameters"
	);

	parameters.DefineIntegerParameter
	(
		"gpu-start-depth",
		"Depth (root=0) at or beyond which subtrees are offloaded to GPU when enabled.",
		3, // default value
		"Algorithmic Parameters",
		0, // min value
		64 // max value
	);

	parameters.DefineIntegerParameter
	(
		"gpu-min-instances",
		"Minimum number of instances in a subproblem before attempting GPU offload.",
		2048, // default value
		"Algorithmic Parameters",
		1, // min value
		100000000 // max value
	);

	parameters.DefineBooleanParameter
	(
		"sort-features-gini-index",
		"Sort features based on Gini index.",
		false,
		"Algorithmic Parameters"
	);

	return parameters;
}