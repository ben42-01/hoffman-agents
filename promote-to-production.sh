WORK_SPACE_DIR=$(pwd)

cd $WORK_SPACE_DIR/hoffman-agents-node && npm publish
cd $WORK_SPACE_DIR/hoffman-agents-python && rm -r dist/* && uv build && uv publish