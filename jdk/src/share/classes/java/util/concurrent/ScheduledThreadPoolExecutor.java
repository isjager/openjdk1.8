/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
 * This file is available under and governed by the GNU General Public
 * License version 2 only, as published by the Free Software Foundation.
 * However, the following notice accompanied the original version of this
 * file:
 *
 * Written by Doug Lea with assistance from members of JCP JSR-166
 * Expert Group and released to the public domain, as explained at
 * http://creativecommons.org/publicdomain/zero/1.0/
 */

package java.util.concurrent;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.*;

/**
 * A {@link ThreadPoolExecutor} that can additionally schedule
 * commands to run after a given delay, or to execute
 * periodically. This class is preferable to {@link java.util.Timer}
 * when multiple worker threads are needed, or when the additional
 * flexibility or capabilities of {@link ThreadPoolExecutor} (which
 * this class extends) are required.
 *
 * <p>Delayed tasks execute no sooner than they are enabled, but
 * without any real-time guarantees about when, after they are
 * enabled, they will commence. Tasks scheduled for exactly the same
 * execution time are enabled in first-in-first-out (FIFO) order of
 * submission.
 *
 * <p>When a submitted task is cancelled before it is run, execution
 * is suppressed. By default, such a cancelled task is not
 * automatically removed from the work queue until its delay
 * elapses. While this enables further inspection and monitoring, it
 * may also cause unbounded retention of cancelled tasks. To avoid
 * this, set {@link #setRemoveOnCancelPolicy} to {@code true}, which
 * causes tasks to be immediately removed from the work queue at
 * time of cancellation.
 *
 * <p>Successive executions of a task scheduled via
 * {@code scheduleAtFixedRate} or
 * {@code scheduleWithFixedDelay} do not overlap. While different
 * executions may be performed by different threads, the effects of
 * prior executions <a
 * href="package-summary.html#MemoryVisibility"><i>happen-before</i></a>
 * those of subsequent ones.
 *
 * <p>While this class inherits from {@link ThreadPoolExecutor}, a few
 * of the inherited tuning methods are not useful for it. In
 * particular, because it acts as a fixed-sized pool using
 * {@code corePoolSize} threads and an unbounded queue, adjustments
 * to {@code maximumPoolSize} have no useful effect. Additionally, it
 * is almost never a good idea to set {@code corePoolSize} to zero or
 * use {@code allowCoreThreadTimeOut} because this may leave the pool
 * without threads to handle tasks once they become eligible to run.
 *
 * <p><b>Extension notes:</b> This class overrides the
 * {@link ThreadPoolExecutor#execute(Runnable) execute} and
 * {@link AbstractExecutorService#submit(Runnable) submit}
 * methods to generate internal {@link ScheduledFuture} objects to
 * control per-task delays and scheduling.  To preserve
 * functionality, any further overrides of these methods in
 * subclasses must invoke superclass versions, which effectively
 * disables additional task customization.  However, this class
 * provides alternative protected extension method
 * {@code decorateTask} (one version each for {@code Runnable} and
 * {@code Callable}) that can be used to customize the concrete task
 * types used to execute commands entered via {@code execute},
 * {@code submit}, {@code schedule}, {@code scheduleAtFixedRate},
 * and {@code scheduleWithFixedDelay}.  By default, a
 * {@code ScheduledThreadPoolExecutor} uses a task type extending
 * {@link FutureTask}. However, this may be modified or replaced using
 * subclasses of the form:
 *
 *  <pre> {@code
 * public class CustomScheduledExecutor extends ScheduledThreadPoolExecutor {
 *
 *   static class CustomTask<V> implements RunnableScheduledFuture<V> { ... }
 *
 *   protected <V> RunnableScheduledFuture<V> decorateTask(
 *                Runnable r, RunnableScheduledFuture<V> task) {
 *       return new CustomTask<V>(r, task);
 *   }
 *
 *   protected <V> RunnableScheduledFuture<V> decorateTask(
 *                Callable<V> c, RunnableScheduledFuture<V> task) {
 *       return new CustomTask<V>(c, task);
 *   }
 *   // ... add constructors, etc.
 * }}</pre>
 *
 * @since 1.5
 * @author Doug Lea
 *
 * 定时线程池
 */
public class ScheduledThreadPoolExecutor
        extends ThreadPoolExecutor
        implements ScheduledExecutorService {

    /*
     * This class specializes ThreadPoolExecutor implementation by
     *
     * 1. Using a custom task type, ScheduledFutureTask for
     *    tasks, even those that don't require scheduling (i.e.,
     *    those submitted using ExecutorService execute, not
     *    ScheduledExecutorService methods) which are treated as
     *    delayed tasks with a delay of zero.
     *
     * 2. Using a custom queue (DelayedWorkQueue), a variant of
     *    unbounded DelayQueue. The lack of capacity constraint and
     *    the fact that corePoolSize and maximumPoolSize are
     *    effectively identical simplifies some execution mechanics
     *    (see delayedExecute) compared to ThreadPoolExecutor.
     *
     * 3. Supporting optional run-after-shutdown parameters, which
     *    leads to overrides of shutdown methods to remove and cancel
     *    tasks that should NOT be run after shutdown, as well as
     *    different recheck logic when task (re)submission overlaps
     *    with a shutdown.
     *
     * 4. Task decoration methods to allow interception and
     *    instrumentation, which are needed because subclasses cannot
     *    otherwise override submit methods to get this effect. These
     *    don't have any impact on pool control logic though.
     */

    /**
     * False if should cancel/suppress periodic tasks on shutdown.
     */
    private volatile boolean continueExistingPeriodicTasksAfterShutdown;

    /**
     * False if should cancel non-periodic tasks on shutdown.
     */
    private volatile boolean executeExistingDelayedTasksAfterShutdown = true;

    /**
     * True if ScheduledFutureTask.cancel should remove from queue
     */
    private volatile boolean removeOnCancel = false;

    /**
     * Sequence number to break scheduling ties, and in turn to
     * guarantee FIFO order among tied entries.
     */
    private static final AtomicLong sequencer = new AtomicLong();

    /**
     * Returns current nanosecond time.
     */
    final long now() {
        return System.nanoTime();
    }

    /**
     * ScheduledFutureTask 类型的任务，是线程池调度任务的最小单位，有三种提交任务的方式：
     * 1、schedule
     * 2、scheduleAtFixedRate
     * 3、scheduleWithFixedDelay
     *
     * 采用 DelayQueue 存储等待的任务：
     * 1、DelayQueue 内部封装了一个 PriorityQueue，它会根据 time 的先后时间排序，若 time 时间
     * 相同则根据 sequenceNumber 排序；
     * 2、DelayQueue 也是一个无界队列；
     */
    private class ScheduledFutureTask<V>
            extends FutureTask<V> implements RunnableScheduledFuture<V> {

        /**
         * Sequence number to break ties FIFO
         *
         * 任务的序号
         */
        private final long sequenceNumber;

        /**
         * The time the task is enabled to execute in nanoTime units
         *
         * 任务开始的时间
         */
        private long time;

        /**
         * Period in nanoseconds for repeating tasks.  A positive
         * value indicates fixed-rate execution.  A negative value
         * indicates fixed-delay execution.  A value of 0 indicates a
         * non-repeating task.
         *
         * 任务执行的时间间隔
         */
        private final long period;

        /** The actual task to be re-enqueued by reExecutePeriodic */
        RunnableScheduledFuture<V> outerTask = this;

        /**
         * Index into delay queue, to support faster cancellation.
         */
        int heapIndex;

        /**
         * Creates a one-shot action with given nanoTime-based trigger time.
         */
        ScheduledFutureTask(Runnable r, V result, long ns) {
            super(r, result);
            this.time = ns;
            this.period = 0;
            this.sequenceNumber = sequencer.getAndIncrement();
        }

        /**
         * Creates a periodic action with given nano time and period.
         */
        ScheduledFutureTask(Runnable r, V result, long ns, long period) {
            super(r, result);
            this.time = ns;
            this.period = period;
            this.sequenceNumber = sequencer.getAndIncrement();
        }

        /**
         * Creates a one-shot action with given nanoTime-based trigger time.
         */
        ScheduledFutureTask(Callable<V> callable, long ns) {
            super(callable);
            this.time = ns;
            this.period = 0;
            this.sequenceNumber = sequencer.getAndIncrement();
        }

        public long getDelay(TimeUnit unit) {
            return unit.convert(time - now(), NANOSECONDS);
        }

        /**
         * 排序算法：
         * DelayQueue 内部封装了一个 PriorityQueue，它会根据 time 的先后时间排序，
         * 若 time 时间相同，则根据 sequenceNumber 排序；
         */
        public int compareTo(Delayed other) {
            if (other == this) // compare zero if same object
                return 0;
            if (other instanceof ScheduledFutureTask) {
                ScheduledFutureTask<?> x = (ScheduledFutureTask<?>)other;
                // 1、先按照 time 排序，time 小的排在前面，time 大的排在后面；
                long diff = time - x.time;
                if (diff < 0)
                    return -1;
                else if (diff > 0)
                    return 1;
                /*
                 * 2、如果 time 相同，按照 sequenceNumber 排序，sequenceNumber 小的排在前面
                 * sequenceNumber 大的排在后面，换句话说，如果两个 task 的执行时间相同，优先执行
                 * 先提交的 task。
                 */
                else if (sequenceNumber < x.sequenceNumber)
                    return -1;
                else
                    return 1;
            }
            long diff = getDelay(NANOSECONDS) - other.getDelay(NANOSECONDS);
            return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
        }

        /**
         * Returns {@code true} if this is a periodic (not a one-shot) action.
         *
         * @return {@code true} if periodic
         */
        public boolean isPeriodic() {
            return period != 0;
        }

        /**
         * Sets the next time to run for a periodic task.
         */
        private void setNextRunTime() {
            long p = period;
            if (p > 0)
                time += p;
            else
                time = triggerTime(-p);
        }

        public boolean cancel(boolean mayInterruptIfRunning) {
            boolean cancelled = super.cancel(mayInterruptIfRunning);
            if (cancelled && removeOnCancel && heapIndex >= 0)
                remove(this);
            return cancelled;
        }

        /**
         * Overrides FutureTask version so as to reset/requeue if periodic.
         *
         * run() 方法是调度 task 的核心，task 的执行实际上是 run() 方法的执行。
         */
        public void run() {
            boolean periodic = isPeriodic();
            // 1、如果当前线程池已经不支持执行任务，则取消该任务，然后直接返回，否则执行步骤2；
            if (!canRunInCurrentRunState(periodic))
                cancel(false);
            /*
             * 2、如果不是周期性任务，调用 FutureTask 中的 run() 方法，会设置执行结果，
             * 然后直接返回，否则执行步骤3；
             */
            else if (!periodic)
                ScheduledFutureTask.super.run();
            /*
             * 3、如果是周期性任务，调用 FutureTask 中的 runAndReset() 方法执行，
             * 不会设置执行结果，然后直接返回，否则执行步骤4和步骤5；
             */
            else if (ScheduledFutureTask.super.runAndReset()) {
                // 4、计算下次执行该任务的具体时间
                setNextRunTime();
                // 5、重复执行任务
                reExecutePeriodic(outerTask);
            }
        }
    }

    /**
     * Returns true if can run a task given current run state
     * and run-after-shutdown parameters.
     *
     * @param periodic true if this task periodic, false if delayed
     */
    boolean canRunInCurrentRunState(boolean periodic) {
        return isRunningOrShutdown(periodic ?
                                   continueExistingPeriodicTasksAfterShutdown :
                                   executeExistingDelayedTasksAfterShutdown);
    }

    /**
     * Main execution method for delayed or periodic tasks.  If pool
     * is shut down, rejects the task. Otherwise adds task to queue
     * and starts a thread, if necessary, to run it.  (We cannot
     * prestart the thread to run the task because the task (probably)
     * shouldn't be run yet.)  If the pool is shut down while the task
     * is being added, cancel and remove it if required by state and
     * run-after-shutdown parameters.
     *
     * delayedExecute：任务提交方法
     *
     * @param task the task
     */
    private void delayedExecute(RunnableScheduledFuture<?> task) {
        // 如果线程池已经关闭，则使用拒绝策略把提交任务拒绝掉
        if (isShutdown())
            reject(task);
        else {
            /*
             * 与 ThreadPoolExecutor 不同，这里直接把任务加入延迟队列，
             * 用的是 DelayedWorkQueue
             */
            super.getQueue().add(task);
            // 如果当前状态无法执行任务，则取消
            if (isShutdown() &&
                !canRunInCurrentRunState(task.isPeriodic()) &&
                remove(task))
                task.cancel(false);
            else
                /*
                 * 这里增加一个 worker 线程，避免提交的任务没有 worker 去执行，
                 * 原因就是该类没有像 ThreadPoolExecutor 一样，worker 满了才放入队列
                 */
                ensurePrestart();
        }
    }

    /**
     * Requeues a periodic task unless current run state precludes it.
     * Same idea as delayedExecute except drops task rather than rejecting.
     *
     * 该方法和 delayedExcetue 方法类似，不同的是：
     * 1、由于调用 reExecutePeriodic() 方法时已经执行过一次周期性任务了，所以不会 reject 当前任务；
     * 2、传入的任务一定时周期性任务；
     *
     * @param task the task
     */
    void reExecutePeriodic(RunnableScheduledFuture<?> task) {
        if (canRunInCurrentRunState(true)) {
            super.getQueue().add(task);
            if (!canRunInCurrentRunState(true) && remove(task))
                task.cancel(false);
            else
                ensurePrestart();
        }
    }

    /**
     * Cancels and clears the queue of all tasks that should not be run
     * due to shutdown policy.  Invoked within super.shutdown.
     */
    @Override void onShutdown() {
        BlockingQueue<Runnable> q = super.getQueue();
        boolean keepDelayed =
            getExecuteExistingDelayedTasksAfterShutdownPolicy();
        boolean keepPeriodic =
            getContinueExistingPeriodicTasksAfterShutdownPolicy();
        if (!keepDelayed && !keepPeriodic) {
            for (Object e : q.toArray())
                if (e instanceof RunnableScheduledFuture<?>)
                    ((RunnableScheduledFuture<?>) e).cancel(false);
            q.clear();
        }
        else {
            // Traverse snapshot to avoid iterator exceptions
            for (Object e : q.toArray()) {
                if (e instanceof RunnableScheduledFuture) {
                    RunnableScheduledFuture<?> t =
                        (RunnableScheduledFuture<?>)e;
                    if ((t.isPeriodic() ? !keepPeriodic : !keepDelayed) ||
                        t.isCancelled()) { // also remove if already cancelled
                        if (q.remove(t))
                            t.cancel(false);
                    }
                }
            }
        }
        tryTerminate();
    }

    /**
     * Modifies or replaces the task used to execute a runnable.
     * This method can be used to override the concrete
     * class used for managing internal tasks.
     * The default implementation simply returns the given task.
     *
     * @param runnable the submitted Runnable
     * @param task the task created to execute the runnable
     * @param <V> the type of the task's result
     * @return a task that can execute the runnable
     * @since 1.6
     */
    protected <V> RunnableScheduledFuture<V> decorateTask(
        Runnable runnable, RunnableScheduledFuture<V> task) {
        return task;
    }

    /**
     * Modifies or replaces the task used to execute a callable.
     * This method can be used to override the concrete
     * class used for managing internal tasks.
     * The default implementation simply returns the given task.
     *
     * @param callable the submitted Callable
     * @param task the task created to execute the callable
     * @param <V> the type of the task's result
     * @return a task that can execute the callable
     * @since 1.6
     */
    protected <V> RunnableScheduledFuture<V> decorateTask(
        Callable<V> callable, RunnableScheduledFuture<V> task) {
        return task;
    }

    /**
     * Creates a new {@code ScheduledThreadPoolExecutor} with the
     * given core pool size.
     *
     * @param corePoolSize the number of threads to keep in the pool, even
     *        if they are idle, unless {@code allowCoreThreadTimeOut} is set
     * @throws IllegalArgumentException if {@code corePoolSize < 0}
     */
    public ScheduledThreadPoolExecutor(int corePoolSize) {
        super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue());
    }

    /**
     * Creates a new {@code ScheduledThreadPoolExecutor} with the
     * given initial parameters.
     *
     * @param corePoolSize the number of threads to keep in the pool, even
     *        if they are idle, unless {@code allowCoreThreadTimeOut} is set
     * @param threadFactory the factory to use when the executor
     *        creates a new thread
     * @throws IllegalArgumentException if {@code corePoolSize < 0}
     * @throws NullPointerException if {@code threadFactory} is null
     */
    public ScheduledThreadPoolExecutor(int corePoolSize,
                                       ThreadFactory threadFactory) {
        super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), threadFactory);
    }

    /**
     * Creates a new ScheduledThreadPoolExecutor with the given
     * initial parameters.
     *
     * @param corePoolSize the number of threads to keep in the pool, even
     *        if they are idle, unless {@code allowCoreThreadTimeOut} is set
     * @param handler the handler to use when execution is blocked
     *        because the thread bounds and queue capacities are reached
     * @throws IllegalArgumentException if {@code corePoolSize < 0}
     * @throws NullPointerException if {@code handler} is null
     */
    public ScheduledThreadPoolExecutor(int corePoolSize,
                                       RejectedExecutionHandler handler) {
        super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), handler);
    }

    /**
     * Creates a new ScheduledThreadPoolExecutor with the given
     * initial parameters.
     *
     * @param corePoolSize the number of threads to keep in the pool, even
     *        if they are idle, unless {@code allowCoreThreadTimeOut} is set
     * @param threadFactory the factory to use when the executor
     *        creates a new thread
     * @param handler the handler to use when execution is blocked
     *        because the thread bounds and queue capacities are reached
     * @throws IllegalArgumentException if {@code corePoolSize < 0}
     * @throws NullPointerException if {@code threadFactory} or
     *         {@code handler} is null
     */
    public ScheduledThreadPoolExecutor(int corePoolSize,
                                       ThreadFactory threadFactory,
                                       RejectedExecutionHandler handler) {
        super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), threadFactory, handler);
    }

    /**
     * Returns the trigger time of a delayed action.
     */
    private long triggerTime(long delay, TimeUnit unit) {
        return triggerTime(unit.toNanos((delay < 0) ? 0 : delay));
    }

    /**
     * Returns the trigger time of a delayed action.
     */
    long triggerTime(long delay) {
        return now() +
            ((delay < (Long.MAX_VALUE >> 1)) ? delay : overflowFree(delay));
    }

    /**
     * Constrains the values of all delays in the queue to be within
     * Long.MAX_VALUE of each other, to avoid overflow in compareTo.
     * This may occur if a task is eligible to be dequeued, but has
     * not yet been, while some other task is added with a delay of
     * Long.MAX_VALUE.
     */
    private long overflowFree(long delay) {
        Delayed head = (Delayed) super.getQueue().peek();
        if (head != null) {
            long headDelay = head.getDelay(NANOSECONDS);
            if (headDelay < 0 && (delay - headDelay < 0))
                delay = Long.MAX_VALUE + headDelay;
        }
        return delay;
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     *
     * 线程池任务的提交：
     * 首先时 schedule() 方法，该方法是指任务在指定延迟时间到达后触发，只会执行一次。
     */
    public ScheduledFuture<?> schedule(Runnable command,
                                       long delay,
                                       TimeUnit unit) {
        // 参数校验
        if (command == null || unit == null)
            throw new NullPointerException();
        /*
         * 这里是一个嵌套结构，首先把用户提交的任务包装成 ScheduledFutureTask 对象，
         * 然后在调用 decorateTask() 进行包装，该方法是留给用户去扩展的，默认是个空方法。
         */
        RunnableScheduledFuture<?> t = decorateTask(command,
            new ScheduledFutureTask<Void>(command, null,
                                          triggerTime(delay, unit)));
        // 包装好任务以后，就进行提交了
        delayedExecute(t);
        return t;
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     */
    public <V> ScheduledFuture<V> schedule(Callable<V> callable,
                                           long delay,
                                           TimeUnit unit) {
        if (callable == null || unit == null)
            throw new NullPointerException();
        RunnableScheduledFuture<V> t = decorateTask(callable,
            new ScheduledFutureTask<V>(callable,
                                       triggerTime(delay, unit)));
        delayedExecute(t);
        return t;
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     * @throws IllegalArgumentException   {@inheritDoc}
     */
    public ScheduledFuture<?> scheduleAtFixedRate(Runnable command,
                                                  long initialDelay,
                                                  long period,
                                                  TimeUnit unit) {
        if (command == null || unit == null)
            throw new NullPointerException();
        if (period <= 0)
            throw new IllegalArgumentException();
        ScheduledFutureTask<Void> sft =
            new ScheduledFutureTask<Void>(command,
                                          null,
                                          triggerTime(initialDelay, unit),
                                          unit.toNanos(period));
        RunnableScheduledFuture<Void> t = decorateTask(command, sft);
        sft.outerTask = t;
        delayedExecute(t);
        return t;
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     * @throws IllegalArgumentException   {@inheritDoc}
     */
    public ScheduledFuture<?> scheduleWithFixedDelay(Runnable command,
                                                     long initialDelay,
                                                     long delay,
                                                     TimeUnit unit) {
        if (command == null || unit == null)
            throw new NullPointerException();
        if (delay <= 0)
            throw new IllegalArgumentException();
        ScheduledFutureTask<Void> sft =
            new ScheduledFutureTask<Void>(command,
                                          null,
                                          triggerTime(initialDelay, unit),
                                          unit.toNanos(-delay));
        RunnableScheduledFuture<Void> t = decorateTask(command, sft);
        sft.outerTask = t;
        delayedExecute(t);
        return t;
    }

    /**
     * Executes {@code command} with zero required delay.
     * This has effect equivalent to
     * {@link #schedule(Runnable,long,TimeUnit) schedule(command, 0, anyUnit)}.
     * Note that inspections of the queue and of the list returned by
     * {@code shutdownNow} will access the zero-delayed
     * {@link ScheduledFuture}, not the {@code command} itself.
     *
     * <p>A consequence of the use of {@code ScheduledFuture} objects is
     * that {@link ThreadPoolExecutor#afterExecute afterExecute} is always
     * called with a null second {@code Throwable} argument, even if the
     * {@code command} terminated abruptly.  Instead, the {@code Throwable}
     * thrown by such a task can be obtained via {@link Future#get}.
     *
     * @throws RejectedExecutionException at discretion of
     *         {@code RejectedExecutionHandler}, if the task
     *         cannot be accepted for execution because the
     *         executor has been shut down
     * @throws NullPointerException {@inheritDoc}
     */
    public void execute(Runnable command) {
        schedule(command, 0, NANOSECONDS);
    }

    // Override AbstractExecutorService methods

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     */
    public Future<?> submit(Runnable task) {
        return schedule(task, 0, NANOSECONDS);
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     */
    public <T> Future<T> submit(Runnable task, T result) {
        return schedule(Executors.callable(task, result), 0, NANOSECONDS);
    }

    /**
     * @throws RejectedExecutionException {@inheritDoc}
     * @throws NullPointerException       {@inheritDoc}
     */
    public <T> Future<T> submit(Callable<T> task) {
        return schedule(task, 0, NANOSECONDS);
    }

    /**
     * Sets the policy on whether to continue executing existing
     * periodic tasks even when this executor has been {@code shutdown}.
     * In this case, these tasks will only terminate upon
     * {@code shutdownNow} or after setting the policy to
     * {@code false} when already shutdown.
     * This value is by default {@code false}.
     *
     * @param value if {@code true}, continue after shutdown, else don't
     * @see #getContinueExistingPeriodicTasksAfterShutdownPolicy
     */
    public void setContinueExistingPeriodicTasksAfterShutdownPolicy(boolean value) {
        continueExistingPeriodicTasksAfterShutdown = value;
        if (!value && isShutdown())
            onShutdown();
    }

    /**
     * Gets the policy on whether to continue executing existing
     * periodic tasks even when this executor has been {@code shutdown}.
     * In this case, these tasks will only terminate upon
     * {@code shutdownNow} or after setting the policy to
     * {@code false} when already shutdown.
     * This value is by default {@code false}.
     *
     * @return {@code true} if will continue after shutdown
     * @see #setContinueExistingPeriodicTasksAfterShutdownPolicy
     */
    public boolean getContinueExistingPeriodicTasksAfterShutdownPolicy() {
        return continueExistingPeriodicTasksAfterShutdown;
    }

    /**
     * Sets the policy on whether to execute existing delayed
     * tasks even when this executor has been {@code shutdown}.
     * In this case, these tasks will only terminate upon
     * {@code shutdownNow}, or after setting the policy to
     * {@code false} when already shutdown.
     * This value is by default {@code true}.
     *
     * @param value if {@code true}, execute after shutdown, else don't
     * @see #getExecuteExistingDelayedTasksAfterShutdownPolicy
     */
    public void setExecuteExistingDelayedTasksAfterShutdownPolicy(boolean value) {
        executeExistingDelayedTasksAfterShutdown = value;
        if (!value && isShutdown())
            onShutdown();
    }

    /**
     * Gets the policy on whether to execute existing delayed
     * tasks even when this executor has been {@code shutdown}.
     * In this case, these tasks will only terminate upon
     * {@code shutdownNow}, or after setting the policy to
     * {@code false} when already shutdown.
     * This value is by default {@code true}.
     *
     * @return {@code true} if will execute after shutdown
     * @see #setExecuteExistingDelayedTasksAfterShutdownPolicy
     */
    public boolean getExecuteExistingDelayedTasksAfterShutdownPolicy() {
        return executeExistingDelayedTasksAfterShutdown;
    }

    /**
     * Sets the policy on whether cancelled tasks should be immediately
     * removed from the work queue at time of cancellation.  This value is
     * by default {@code false}.
     *
     * @param value if {@code true}, remove on cancellation, else don't
     * @see #getRemoveOnCancelPolicy
     * @since 1.7
     */
    public void setRemoveOnCancelPolicy(boolean value) {
        removeOnCancel = value;
    }

    /**
     * Gets the policy on whether cancelled tasks should be immediately
     * removed from the work queue at time of cancellation.  This value is
     * by default {@code false}.
     *
     * @return {@code true} if cancelled tasks are immediately removed
     *         from the queue
     * @see #setRemoveOnCancelPolicy
     * @since 1.7
     */
    public boolean getRemoveOnCancelPolicy() {
        return removeOnCancel;
    }

    /**
     * Initiates an orderly shutdown in which previously submitted
     * tasks are executed, but no new tasks will be accepted.
     * Invocation has no additional effect if already shut down.
     *
     * <p>This method does not wait for previously submitted tasks to
     * complete execution.  Use {@link #awaitTermination awaitTermination}
     * to do that.
     *
     * <p>If the {@code ExecuteExistingDelayedTasksAfterShutdownPolicy}
     * has been set {@code false}, existing delayed tasks whose delays
     * have not yet elapsed are cancelled.  And unless the {@code
     * ContinueExistingPeriodicTasksAfterShutdownPolicy} has been set
     * {@code true}, future executions of existing periodic tasks will
     * be cancelled.
     *
     * @throws SecurityException {@inheritDoc}
     */
    public void shutdown() {
        super.shutdown();
    }

    /**
     * Attempts to stop all actively executing tasks, halts the
     * processing of waiting tasks, and returns a list of the tasks
     * that were awaiting execution.
     *
     * <p>This method does not wait for actively executing tasks to
     * terminate.  Use {@link #awaitTermination awaitTermination} to
     * do that.
     *
     * <p>There are no guarantees beyond best-effort attempts to stop
     * processing actively executing tasks.  This implementation
     * cancels tasks via {@link Thread#interrupt}, so any task that
     * fails to respond to interrupts may never terminate.
     *
     * @return list of tasks that never commenced execution.
     *         Each element of this list is a {@link ScheduledFuture},
     *         including those tasks submitted using {@code execute},
     *         which are for scheduling purposes used as the basis of a
     *         zero-delay {@code ScheduledFuture}.
     * @throws SecurityException {@inheritDoc}
     */
    public List<Runnable> shutdownNow() {
        return super.shutdownNow();
    }

    /**
     * Returns the task queue used by this executor.  Each element of
     * this queue is a {@link ScheduledFuture}, including those
     * tasks submitted using {@code execute} which are for scheduling
     * purposes used as the basis of a zero-delay
     * {@code ScheduledFuture}.  Iteration over this queue is
     * <em>not</em> guaranteed to traverse tasks in the order in
     * which they will execute.
     *
     * @return the task queue
     */
    public BlockingQueue<Runnable> getQueue() {
        return super.getQueue();
    }

    /**
     * Specialized delay queue. To mesh with TPE declarations, this
     * class must be declared as a BlockingQueue<Runnable> even though
     * it can only hold RunnableScheduledFutures.
     *
     * 专门的延迟队列。为了 TPE 声明相吻合，这即使该类只能容纳 RunnableScheduledFutures，
     * 也必须将其声明为 BlockingQueue<Runnable>
     *
     * ScheduledThreadPoolExecutor 只所以要自己实现阻塞的工作队列，是因为 ScheduledThreadPoolExecutor 要求的工作队列有些特殊。
     *
     * DelayedWorkQueue 是一个基于堆的数据结构，类似于 DelayQueue 和 PriorityQueue。
     * 在执行定时任务的时候，每个任务的执行时间都不同，所以 DelayedWorkQueue 的工作就是
     * 按照执行时间的升序来排序，执行时间距离当前时间越近的任务在队列的前面。
     * （注意：这里的顺序并不是绝对的，堆中的排序只保证了子节点的下次执行时间要比父节点的下次执行
     * 时间要大，而叶子节点之间并不一定是顺序的）
     *
     * 堆结构的特性：
     * 假设，索引值从0开始，子节点的索引值为k，父节点的索引值为p，则：
     * 1、一个节点的左子节点的索引为：k = p * 2 + 1；
     * 2、一个节点的右子节点的索引为：k = (p + 1) * 2；
     * 3、一个节点的父节点的索引为：p = (k -1)/2
     *
     * 为什么要使用 DelayedWorkQueue 呢？
     *     定时任务执行时需要取出最近要执行的任务，所以任务在队列中每次出队时一定要是当前队列中
     * 执行时间最靠前的，所以自然要使用优先级队列。
     *      DelayedWorkQueue 是一个优先级队列，它可以保证每次出队的任务都是当前队列中执行时间
     * 最靠前的，由于它是基于堆结构的队列，堆结构在执行插入和删除操作时的最坏时间复杂度是 O(logN)。
     */
    static class DelayedWorkQueue extends AbstractQueue<Runnable>
        implements BlockingQueue<Runnable> {

        /*
         * A DelayedWorkQueue is based on a heap-based data structure
         * like those in DelayQueue and PriorityQueue, except that
         * every ScheduledFutureTask also records its index into the
         * heap array. This eliminates the need to find a task upon
         * cancellation, greatly speeding up removal (down from O(n)
         * to O(log n)), and reducing garbage retention that would
         * otherwise occur by waiting for the element to rise to top
         * before clearing. But because the queue may also hold
         * RunnableScheduledFutures that are not ScheduledFutureTasks,
         * we are not guaranteed to have such indices available, in
         * which case we fall back to linear search. (We expect that
         * most tasks will not be decorated, and that the faster cases
         * will be much more common.)
         *
         * All heap operations must record index changes -- mainly
         * within siftUp and siftDown. Upon removal, a task's
         * heapIndex is set to -1. Note that ScheduledFutureTasks can
         * appear at most once in the queue (this need not be true for
         * other kinds of tasks or work queues), so are uniquely
         * identified by heapIndex.
         */

        // 队列初始容量
        private static final int INITIAL_CAPACITY = 16;
        // 根据初始容量创建 RunnableScheduledFuture 类型的数组
        private RunnableScheduledFuture<?>[] queue =
            new RunnableScheduledFuture<?>[INITIAL_CAPACITY];
        private final ReentrantLock lock = new ReentrantLock();
        private int size = 0;

        /**
         * Thread designated to wait for the task at the head of the
         * queue.  This variant of the Leader-Follower pattern
         * (http://www.cs.wustl.edu/~schmidt/POSA/POSA2/) serves to
         * minimize unnecessary timed waiting.  When a thread becomes
         * the leader, it waits only for the next delay to elapse, but
         * other threads await indefinitely.  The leader thread must
         * signal some other thread before returning from take() or
         * poll(...), unless some other thread becomes leader in the
         * interim.  Whenever the head of the queue is replaced with a
         * task with an earlier expiration time, the leader field is
         * invalidated by being reset to null, and some waiting
         * thread, but not necessarily the current leader, is
         * signalled.  So waiting threads must be prepared to acquire
         * and lose leadership while waiting.
         *
         * leader 线程
         */
        private Thread leader = null;

        /**
         * Condition signalled when a newer task becomes available at the
         * head of the queue or a new thread may need to become leader.
         *
         * 当较新的任务在队列的头部可用时，或者新线程可能需要成为 leader，则通过该条件发出信号
         *
         * leader 是 Leader-Follower 模式的变体，用于减少不必要的定时等待。什么意思呢？
         * 对于多线程的模型来说：
         *     1、所有线程会有三种身份中的一种：leader 和 follower，以及一个干活中的状态：proccesser。
         * 它的基本原则就是，永远最多只有一个 leader。而所有 follower 都在等待成为 leader。
         *     2、线程池启动时会自动产生一个 Leader 负责等待网络 IO 事件，当有一个事件产生时，Leader 线程
         * 首先通知一个 Follower 线程将其提拔为新 Leader，然后自己就去干活了，去处理这个网络事件，处理完毕
         * 后加入 Follower 线程等待队列，等待下次成为 Leader。这种方法可以增强 CPU 高速缓存相似性，及消除
         * 动态内存分配和线程间的数据交换。
         */
        private final Condition available = lock.newCondition();

        /**
         * Sets f's heapIndex if it is a ScheduledFutureTask.
         */
        private void setIndex(RunnableScheduledFuture<?> f, int idx) {
            if (f instanceof ScheduledFutureTask)
                ((ScheduledFutureTask)f).heapIndex = idx;
        }

        /**
         * Sifts element added at bottom up to its heap-ordered spot.
         * Call only when holding lock.
         *
         * 此处代码，就是循环的根据 key 节点与它的父节点来判断，如果 key 节点的执行时间小于父节点，
         * 则将两个节点交换，使执行时间靠前的节点排列在队列的前面。
         */
        private void siftUp(int k, RunnableScheduledFuture<?> key) {
            // 找到父节点的索引
            while (k > 0) {
                // 获取父节点
                int parent = (k - 1) >>> 1;
                RunnableScheduledFuture<?> e = queue[parent];
                // 如果 key 节点的执行时间大于父节点的执行时间，不需要在排序了
                if (key.compareTo(e) >= 0)
                    break;
                // 如果 key.compareTo(e) < 0，说明 key 节点的执行时间小于父节点的执行时间，需要把父节点移到后面
                queue[k] = e;
                setIndex(e, k);
                // 设置索引为 k
                k = parent;
            }
            // key 设置为排序后的位置中
            queue[k] = key;
            setIndex(key, k);
        }

        /**
         * Sifts element added at top down to its heap-ordered spot.
         * Call only when holding lock.
         *
         * siftDown() 方法，使堆从 k 开始向下调整
         *
         * siftDown() 方法执行时包含两种情况：
         * 1、一种是没有子节点，
         * 2、一种是有子节点（根据 half 判断）
         */
        private void siftDown(int k, RunnableScheduledFuture<?> key) {
            // 根据二叉树的特性，数组长度除以2，表示取有子节点的索引
            int half = size >>> 1;
            // 判断索引为 k 的节点是否有子节点
            while (k < half) {
                // 左子节点的索引
                int child = (k << 1) + 1;
                RunnableScheduledFuture<?> c = queue[child];
                // 右子节点的索引
                int right = child + 1;
                // 如果有右子节点并且左子节点的时间间隔大于右子节点，取时间间隔最小的节点
                if (right < size && c.compareTo(queue[right]) > 0)
                    c = queue[child = right];
                // 如果 key 的时间间隔小于等于 c 的时间间隔，跳出循环
                if (key.compareTo(c) <= 0)
                    break;
                // 设置要移除索引的节点为其子节点
                queue[k] = c;
                setIndex(c, k);
                k = child;
            }
            //将 key 放入索引为 k 的位置
            queue[k] = key;
            setIndex(key, k);
        }

        /**
         * Resizes the heap array.  Call only when holding lock.
         */
        private void grow() {
            int oldCapacity = queue.length;
            int newCapacity = oldCapacity + (oldCapacity >> 1); // grow 50%
            if (newCapacity < 0) // overflow
                newCapacity = Integer.MAX_VALUE;
            queue = Arrays.copyOf(queue, newCapacity);
        }

        /**
         * Finds index of given object, or -1 if absent.
         */
        private int indexOf(Object x) {
            if (x != null) {
                if (x instanceof ScheduledFutureTask) {
                    int i = ((ScheduledFutureTask) x).heapIndex;
                    // Sanity check; x could conceivably be a
                    // ScheduledFutureTask from some other pool.
                    if (i >= 0 && i < size && queue[i] == x)
                        return i;
                } else {
                    for (int i = 0; i < size; i++)
                        if (x.equals(queue[i]))
                            return i;
                }
            }
            return -1;
        }

        public boolean contains(Object x) {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                return indexOf(x) != -1;
            } finally {
                lock.unlock();
            }
        }

        public boolean remove(Object x) {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                int i = indexOf(x);
                if (i < 0)
                    return false;

                setIndex(queue[i], -1);
                int s = --size;
                RunnableScheduledFuture<?> replacement = queue[s];
                queue[s] = null;
                if (s != i) {
                    // 从 i 开始向下调整
                    siftDown(i, replacement);
                    /*
                     * 如果 queue[i] == replacement，说明 i 是叶子节点
                     * 如果是这种情况，不能保证子节点的下次执行时间比父节点的大
                     * 这时需要进行一次向上调整
                     */
                    if (queue[i] == replacement)
                        siftUp(i, replacement);
                }
                return true;
            } finally {
                lock.unlock();
            }
        }

        public int size() {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                return size;
            } finally {
                lock.unlock();
            }
        }

        public boolean isEmpty() {
            return size() == 0;
        }

        public int remainingCapacity() {
            return Integer.MAX_VALUE;
        }

        public RunnableScheduledFuture<?> peek() {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                return queue[0];
            } finally {
                lock.unlock();
            }
        }

        public boolean offer(Runnable x) {
            // 参数校验
            if (x == null)
                throw new NullPointerException();
            RunnableScheduledFuture<?> e = (RunnableScheduledFuture<?>)x;
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                // 查看当前元素数量，如果大于队列长度则进行扩容
                int i = size;
                if (i >= queue.length)
                    // 扩容
                    grow();
                // 元素数量加 1
                size = i + 1;
                if (i == 0) {
                    queue[0] = e;
                    // 记录索引
                    setIndex(e, 0);
                } else {
                    /*
                     * 把任务加入堆中，并调整堆结构，这里就会根据任务的触发时间排列，
                     * 把需要最早执行的任务放在前面
                     */
                    siftUp(i, e);
                }
                /*
                 * 如果新加入的元素就是队列头，这里有两种情况：
                 * 1、这是用户提交的第一个任务
                 * 2、新任务进行堆调整后，排在队列头
                 */
                if (queue[0] == e) {
                    /*
                     * leader 设置为 null，为了使在 task() 方法中的线程在通过 available.signal()；
                     * 后会执行 available.awaitNanos(delay)
                     */
                    leader = null;
                    // 加入元素以后，唤醒 worker 线程
                    available.signal();
                }
            } finally {
                lock.unlock();
            }
            return true;
        }

        public void put(Runnable e) {
            offer(e);
        }

        public boolean add(Runnable e) {
            return offer(e);
        }

        public boolean offer(Runnable e, long timeout, TimeUnit unit) {
            return offer(e);
        }

        /**
         * Performs common bookkeeping for poll and take: Replaces
         * first element with last and sifts it down.  Call only when
         * holding lock.
         *
         * 当调用了 take() 或 poll() 方法能够获取任务时，会调用该方法进行返回
         *
         * @param f the task to remove and return
         */
        private RunnableScheduledFuture<?> finishPoll(RunnableScheduledFuture<?> f) {
            // 数组长度-1
            int s = --size;
            // 取出最后一个节点
            RunnableScheduledFuture<?> x = queue[s];
            queue[s] = null;
            // 长度不为0，则从这一个元素开始排序，目的是要把最后一个节点放到合适的位置上
            if (s != 0)
                siftDown(0, x);
            setIndex(f, -1);
            return f;
        }

        /**
         * poll() 方法，也 take() 方法类似，这里要提供超时功能：
         *
         * @return
         */
        public RunnableScheduledFuture<?> poll() {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                RunnableScheduledFuture<?> first = queue[0];
                if (first == null || first.getDelay(NANOSECONDS) > 0)
                    return null;
                else
                    return finishPoll(first);
            } finally {
                lock.unlock();
            }
        }

        /**
         * take() 方法是什么时候调用呢？在 ThreadPoolExecutor 中，介绍了 getTask() 方法，
         * 工作线程会循环地从 workQueue 中取任务。但定时任务却不同，因为如果一旦 getTask() 方法
         * 取出了任务就开始执行了，而这时可能还没有到执行的时间，所以 take() 方法中，要保证只有在
         * 指定的执行时间的时候任务才可以被取走。
         *
         * leader 的作用，这里的 leader 是为了减少不必要的定时等待，当一个线程成为 leader 时，它
         * 只等待下一个节点的时间间隔，但其他线程无限期等待。leader 线程必须在 take() 或 poll()
         * 返回之前 signal 其他线程，除非其他线程成为了 leader。
         *
         * 例：
         * 如果没有 leader，那么在执行 take() 方法时，都要执行 available.awaitNanos(delay); 假设
         * 当前线程执行了该段代码，这时还没有 signal，第二个线程也执行了该段代码，则第二个线程也要被阻塞。
         * 多个这时执行该段代码是没有作用的，因为只能有一个线程会 take() 方法中返回 queue[0]（因为有 lock），
         * 其他线程这时再返回 for 循环执行时取 queue[0]，已经不是之前的 queue[0] 了，然后又要继续阻塞。
         *
         * 所以，为了不让多个线程频繁的做无用的定时等待，这里增加了 leader，如果 leader 不为空，则说明队列
         * 中第一个节点已经在等待出队，这时其他的线程会一直阻塞，减少了无用的阻塞（注意，finally 中调用了
         * signal() 来唤醒一个线程，而不是 signalAll()）
         *
         * @return
         * @throws InterruptedException
         */
        public RunnableScheduledFuture<?> take() throws InterruptedException {
            final ReentrantLock lock = this.lock;
            lock.lockInterruptibly();
            try {
                for (;;) {
                    RunnableScheduledFuture<?> first = queue[0];
                    if (first == null)
                        available.await();
                    else {
                        // 计算当前时间到执行时间的间隔时间
                        long delay = first.getDelay(NANOSECONDS);
                        if (delay <= 0)
                            return finishPoll(first);
                        first = null; // don't retain ref while waiting
                        // leader 不为空，阻塞线程
                        if (leader != null)
                            available.await();
                        else {
                            // leader 为空，则把 leader 设置为当前线程
                            Thread thisThread = Thread.currentThread();
                            leader = thisThread;
                            try {
                                // 阻塞到执行时间
                                available.awaitNanos(delay);
                            } finally {
                                // 设置 leader = null，让其他线程执行 available.awaitNanos(delay)
                                if (leader == thisThread)
                                    leader = null;
                            }
                        }
                    }
                }
            } finally {
                /*
                 * 如果 leader 不为空，则说明 leader 的线程正在执行 available.awaitNanos(delay);
                 *
                 * 如果 queue[0] == null，说明队列为空
                 */
                if (leader == null && queue[0] != null)
                    //  Wakes up one waiting thread.
                    available.signal();
                lock.unlock();
            }
        }

        public RunnableScheduledFuture<?> poll(long timeout, TimeUnit unit)
            throws InterruptedException {
            long nanos = unit.toNanos(timeout);
            final ReentrantLock lock = this.lock;
            lock.lockInterruptibly();
            try {
                for (;;) {
                    RunnableScheduledFuture<?> first = queue[0];
                    if (first == null) {
                        if (nanos <= 0)
                            return null;
                        else
                            nanos = available.awaitNanos(nanos);
                    } else {
                        long delay = first.getDelay(NANOSECONDS);
                        if (delay <= 0)
                            return finishPoll(first);
                        if (nanos <= 0)
                            return null;
                        first = null; // don't retain ref while waiting
                        if (nanos < delay || leader != null)
                            nanos = available.awaitNanos(nanos);
                        else {
                            Thread thisThread = Thread.currentThread();
                            leader = thisThread;
                            try {
                                long timeLeft = available.awaitNanos(delay);
                                nanos -= delay - timeLeft;
                            } finally {
                                if (leader == thisThread)
                                    leader = null;
                            }
                        }
                    }
                }
            } finally {
                if (leader == null && queue[0] != null)
                    available.signal();
                lock.unlock();
            }
        }

        public void clear() {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                for (int i = 0; i < size; i++) {
                    RunnableScheduledFuture<?> t = queue[i];
                    if (t != null) {
                        queue[i] = null;
                        setIndex(t, -1);
                    }
                }
                size = 0;
            } finally {
                lock.unlock();
            }
        }

        /**
         * Returns first element only if it is expired.
         * Used only by drainTo.  Call only when holding lock.
         */
        private RunnableScheduledFuture<?> peekExpired() {
            // assert lock.isHeldByCurrentThread();
            RunnableScheduledFuture<?> first = queue[0];
            return (first == null || first.getDelay(NANOSECONDS) > 0) ?
                null : first;
        }

        public int drainTo(Collection<? super Runnable> c) {
            if (c == null)
                throw new NullPointerException();
            if (c == this)
                throw new IllegalArgumentException();
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                RunnableScheduledFuture<?> first;
                int n = 0;
                while ((first = peekExpired()) != null) {
                    c.add(first);   // In this order, in case add() throws.
                    finishPoll(first);
                    ++n;
                }
                return n;
            } finally {
                lock.unlock();
            }
        }

        public int drainTo(Collection<? super Runnable> c, int maxElements) {
            if (c == null)
                throw new NullPointerException();
            if (c == this)
                throw new IllegalArgumentException();
            if (maxElements <= 0)
                return 0;
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                RunnableScheduledFuture<?> first;
                int n = 0;
                while (n < maxElements && (first = peekExpired()) != null) {
                    c.add(first);   // In this order, in case add() throws.
                    finishPoll(first);
                    ++n;
                }
                return n;
            } finally {
                lock.unlock();
            }
        }

        public Object[] toArray() {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                return Arrays.copyOf(queue, size, Object[].class);
            } finally {
                lock.unlock();
            }
        }

        @SuppressWarnings("unchecked")
        public <T> T[] toArray(T[] a) {
            final ReentrantLock lock = this.lock;
            lock.lock();
            try {
                if (a.length < size)
                    return (T[]) Arrays.copyOf(queue, size, a.getClass());
                System.arraycopy(queue, 0, a, 0, size);
                if (a.length > size)
                    a[size] = null;
                return a;
            } finally {
                lock.unlock();
            }
        }

        public Iterator<Runnable> iterator() {
            return new Itr(Arrays.copyOf(queue, size));
        }

        /**
         * Snapshot iterator that works off copy of underlying q array.
         */
        private class Itr implements Iterator<Runnable> {
            final RunnableScheduledFuture<?>[] array;
            int cursor = 0;     // index of next element to return
            int lastRet = -1;   // index of last element, or -1 if no such

            Itr(RunnableScheduledFuture<?>[] array) {
                this.array = array;
            }

            public boolean hasNext() {
                return cursor < array.length;
            }

            public Runnable next() {
                if (cursor >= array.length)
                    throw new NoSuchElementException();
                lastRet = cursor;
                return array[cursor++];
            }

            public void remove() {
                if (lastRet < 0)
                    throw new IllegalStateException();
                DelayedWorkQueue.this.remove(array[lastRet]);
                lastRet = -1;
            }
        }
    }
}
