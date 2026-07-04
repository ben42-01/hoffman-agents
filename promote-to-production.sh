WORK_SPACE_DIR = $(shell pwd)

cd $WORK_SPACE_DIR/hoffman-agents-node && npm version patch && npm publish
cd $WORK_SPACE_DIR/hoffman-agents-python && rm -r dist/* && uv build && uv publish