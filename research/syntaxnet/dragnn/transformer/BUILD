py_binary(
    name = "make_parser_spec",
    srcs = ["make_parser_spec.py"],
    deps = [
        "//dragnn/protos:spec_py_pb2",
        "//dragnn/python:spec_builder",
        "//syntaxnet:dictionary_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "convert_embeddings_for_syntaxnet",
    srcs = ["convert_embeddings_for_syntaxnet.py"],
    deps = [
        "//syntaxnet:dictionary_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
