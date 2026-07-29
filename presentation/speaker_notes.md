# FlameGuard AI - Speaker Notes

Target length: ~15 minutes (planned 21m 0s across 15 slides, leaving room for questions).

## Slide 1 - Title  (0:45)

Open: 'Fire kills, and the clock starts at ignition - not at alarm.' Introduce the team and the one-line goal: detect fire and smoke in images, video and live camera, using a model we fine-tuned ourselves. Keep this to 45 seconds.

## Slide 2 - The problem  (1:15)

Make the motivation concrete: a smoke detector needs the smoke to physically reach it, which in a warehouse, an atrium or a forest can take minutes or never happen. Cameras already watch these spaces. The hard part is that smoke looks like fog, steam and cloud, and sunsets look like fire - so the model must balance missing a real fire against crying wolf.

## Slide 3 - Objective and scope  (1:00)

State scope crisply. Two classes, transfer learning (the course forbids presenting an unchanged pretrained model), a real application, and honest evaluation. Mention the constraint that shaped everything: one 4GB laptop GPU.

## Slide 4 - The dataset  (1:00)

Give the shape of the data: 5,300 images, 7,225 boxes, imbalanced roughly 2.3 to 1 toward fire. Emphasise the 2,076 background images - they are not waste, they are how the model learns not to alarm on a sunset.

## Slide 5 - What we found before training anything  (2:00)

This is the slide to slow down on. The dataset ships mirrored and noise-injected copies of its own images, and it scattered them across train, validation and test. 409 groups - 64% of all images - had members in more than one split. Training on that and reporting test numbers would measure memorisation. We grouped images by perceptual hash and rebuilt the splits group-wise. Every number in this talk comes from the repaired data. If you take one thing from this presentation, take this one.

## Slide 6 - What the data told us  (1:15)

Three findings that changed decisions: 15% of boxes are small, so we kept the full 640px input instead of downscaling for speed. Smoke sits high in the frame and fire sits low - which is why we disabled vertical flip augmentation: an upside-down plume is not a scene that exists. And a tenth of the images are dark night fires, so we kept brightness jitter moderate.

## Slide 7 - The model  (1:30)

Explain transfer learning in one breath: COCO has 80 classes and none of them is fire, so the pretrained model cannot detect fire at all. What it brings is generic vision - edges, textures, shapes - learned from 118,000 images. We fine-tune ALL layers on our data. Single-stage detection is what makes the live webcam demo possible.

## Slide 8 - Experiments  (1:30)

Be honest about the compute wall: FP32 was forced on us because AMP produces NaN losses on GTX 16xx GPUs. We measured YOLOv8s's memory cost before training it - batch 8 wants ~7.9GB, batch 4 ~6.1GB, both spilling to system RAM - so we trained it at batch 2, the only size that fits, with gradient accumulation to a nominal batch of 64. It completed, and the bigger backbone still did not beat the baseline at our budget.

## Slide 9 - Hyperparameter tuning  (1:15)

The design matters more than the result: a control run at the same budget, then three probes each changing exactly one thing. That is what makes any difference attributable. Report the outcome honestly - including probes that did not beat the control.

## Slide 10 - Final results  (1:45)

Stress the discipline: the threshold was chosen on validation, then the test set was touched once. Overall mAP@0.5 is 0.505. The per-class gap is the story - Fire AP 0.513 vs Smoke 0.498. Smoke is harder, exactly as the EDA predicted.

## Slide 11 - Where it fails  (1:45)

Do not skip this. The model makes almost no class confusions (1 in the whole test set) - its errors are about whether something is there at all. Then land the punchline: we fed it a plain orange rectangle - no fire, no texture, no structure - and it said Fire with 0.00 confidence. The model has learned a colour prior, not a concept of flame. That one experiment explains the entire false-positive gallery, and it is why hard-negative mining beats architecture search as the next step.

## Slide 12 - The application  (1:15)

Point out the engineering: image, video and webcam all call the same predict path, so a threshold means the same thing everywhere and there is exactly one place a detection bug could hide. Mention the OpenCV fallback and why it exists - browser camera access fails in exactly the situation where a demo must not fail.

## Slide 13 - Live demonstration  (2:30)

DEMO ORDER (rehearse this): 1) show model + device status in the sidebar; 2) upload a fire image; 3) upload a smoke image; 4) upload a both-classes image; 5) upload a negative/difficult image and let it be wrong - explain why; 6) run the short video; 7) start the webcam; 8) show the performance tab. If the webcam fails, switch to the OpenCV fallback; if that fails, play the backup video in presentation/backup_demo/. Never debug live - fall back.

## Slide 14 - How we worked  (1:00)

Keep it short. The point worth making: the risk register earned its keep - we identified the leakage risk during Sprint 1 planning and mitigated it before spending GPU time on a model whose evaluation would have been worthless.

## Slide 15 - Conclusion and future work  (1:15)

Land the three lessons. Then the disclaimer - say it out loud, do not just leave it on the slide. Finish by inviting questions.

## If something goes wrong

- Webcam will not start → switch to `python src/webcam_inference.py`.
- That also fails → play `presentation/backup_demo/` video and screenshots.
- Video processing is slow → set frame skip to 'every 3rd frame' first.
- Never debug live. Narrate the fallback and keep moving.
