.PHONY: app offline vis_det setup

# Gradio demo (remote workspace)
app:
	export GRADIO_ROOT_PATH=/ws-6514d757-318d-4f55-b2e7-a9a663c75632/project-9b026db4-8fd2-4e8f-b211-79b4db5bed44/user-c002651d-725c-4413-a6b8-736d902e3f65/vscode/6b811c18-82df-4292-ae94-f96e2d1326b0/3a71d545-cf3c-41df-a53b-9b08b4db2534/proxy/7860/ && \
	export PYTHONPATH="/inspire/hdd/global_user/xitong-inspire-admin/sam-body4d/models/sam3:$$PYTHONPATH" && \
	export PYOPENGL_PLATFORM=osmesa && \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
	python app.py

# Offline pipeline
offline:
	export PYTHONPATH="/inspire/hdd/global_user/xitong-inspire-admin/sam-body4d/models/sam3:$$PYTHONPATH" && \
	export PYOPENGL_PLATFORM=osmesa && \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
	python scripts/offline_app.py $(ARGS)

# Visualize detections (save bbox image then exit)
vis_det:
	export PYTHONPATH="/inspire/hdd/global_user/xitong-inspire-admin/sam-body4d/models/sam3:$$PYTHONPATH" && \
	export PYOPENGL_PLATFORM=osmesa && \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
	python scripts/offline_app.py --vis_det $(ARGS)

# Download checkpoints
setup:
	python scripts/setup.py --ckpt-root $(CKPT_ROOT)
