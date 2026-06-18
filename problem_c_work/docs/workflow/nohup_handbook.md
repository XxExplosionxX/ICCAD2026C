# Nohup Handbook

## What `nohup` does

`nohup` lets a command keep running after you close the SSH session.

Typical use:
- start training on the Linux server
- disconnect from SSH
- reconnect later
- check the log file

## Basic pattern

Run a command in the background and save all output to a log file:

```tcsh
nohup COMMAND >& LOGFILE &
```

For `tcsh`, use:
- `>& LOGFILE` to redirect both stdout and stderr
- `&` to run in background

## Example for this project

If you are in:

```tcsh
/path/to/cad2026/problem_c_work
```

run:

```tcsh
nohup /path/to/python -u active/scripts/ml/train_m7_center.py --data-path /path/to/repo-root --epochs 20 --batch-size 32 --train-samples 16384 --val-samples 256 --checkpoint-dir artifacts/checkpoints/ml --result-dir artifacts/results/ml --log-interval 10 --run-name m7_center_scale1 >& artifacts/results/ml/m7_center_scale1.log &
```

## What the output means

After starting the command, you may see something like:

```text
[1] 10181
```

Meaning:
- `[1]` = shell job number
- `10181` = process ID (`PID`)

The important part is the `PID`.

## How to see the output

Watch the log file:

```tcsh
tail -f artifacts/results/ml/m7_center_scale1.log
```

Stop following with:
- `Ctrl+C`

This does **not** stop the training job. It only stops the log viewer.

## How to check whether the job is still running

If you know the PID:

```tcsh
ps -p 10181 -f
```

Or search by script name:

```tcsh
ps -fu $USER | grep train_m7_center.py
```

## How to check GPU usage

```tcsh
nvidia-smi
```

Look for:
- your Python process
- GPU memory usage
- utilization percentage

## How to stop the job

Use the PID:

```tcsh
kill 10181
```

If the job does not stop:

```tcsh
kill -9 10181
```

## Recommended workflow

1. SSH into server
2. Go to the project folder
3. Start training with `nohup`
4. Check log with `tail -f`
5. Disconnect safely
6. Reconnect later
7. Check log and output files

## Common mistakes

### 1. Wrong Python

Bad:

```tcsh
nohup python ...
```

if `python` points to the old system Python.

Better:

```tcsh
nohup /path/to/python ...
```

### 2. Wrong redirect syntax in `tcsh`

Bad:

```tcsh
> file 2>&1 &
```

That is shell syntax for `bash`, not `tcsh`.

Good:

```tcsh
>& file &
```

### 3. Wrong working directory

If you are already inside `problem_c_work`, do:

```tcsh
scripts/ml/train_m7_center.py
```

not:

```tcsh
active/scripts/ml/train_m7_center.py
```

## Quick commands

Start:

```tcsh
nohup /path/to/python -u active/scripts/ml/train_m7_center.py --data-path /path/to/repo-root --epochs 20 --batch-size 32 --train-samples 16384 --val-samples 256 --checkpoint-dir artifacts/checkpoints/ml --result-dir artifacts/results/ml --log-interval 10 --run-name m7_center_scale1 >& artifacts/results/ml/m7_center_scale1.log &
```

Watch log:

```tcsh
tail -f artifacts/results/ml/m7_center_scale1.log
```

Check process:

```tcsh
ps -fu $USER | grep train_m7_center.py
```

Check GPU:

```tcsh
nvidia-smi
```

Stop process:

```tcsh
kill PID
```
