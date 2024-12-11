package algorithm.BacktrackSTLUtils.common;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * @author Haoyu Wang
 * @date 2022/7/29
 */

public class CircularQueue<E> {

    private E[] data;

    private int start = 0;

    private int size = 0;

    private int capacity = 0;

    public CircularQueue() {
    }

    /**
     * Create a circular queue with certain capacity.
     *
     * @param capacity Capacity of the circular queue.
     */
    public CircularQueue(int capacity) {
        this.capacity = capacity;
        this.data = (E[])new Object[capacity];
    }

    /**
     * Insert the specified element into the tail of the circular queue. If the queue is at full capacity,
     * the first element will be overwritten.
     *
     * @param e The inserted element
     */
    public void add(E e) {
        if (isAtFullCapacity()) {
            data[start] = e;
            start = (start + 1) % capacity;
        } else {
            data[(start + size) % capacity] = e;
            size++;
        }
    }

    /**
     * Get the element at the specified index.
     *
     * @param index Index of the element
     * @return the element at the specified index
     */
    public E get(int index) {
        if (index < 0 || index >= size) {
            throw new NoSuchElementException(
                String.format("The specified index (%1$d) is outside the available range [0, %2$d)", index, size));
        }
        final int idx = (start + index) % capacity;
        return data[idx];
    }

    /**
     * Get the first element of the circular queue.
     *
     * @return the first element
     */
    public E getFirst() {
        if (size == 0) {
            throw new NoSuchElementException("The circular queue is empty thus not able to get the first element");
        }
        return data[start];
    }

    /**
     * Get the last element of the circular queue.
     *
     * @return the last element
     */
    public E getLast() {
        if (size == 0) {
            throw new NoSuchElementException("The circular queue is empty thus not able to get the last element");
        }
        return data[(start + size - 1) % capacity];
    }

    /**
     * Delete the first element of the circular queue.
     *
     * @return the deleted first element
     */
    public E removeFirst() {
        if (size == 0) {
            throw new NoSuchElementException("The circular queue is empty thus not able to remove the first element");
        }
        E e = data[start];
        data[start] = null;
        start = (start + 1) % capacity;
        size--;
        return e;
    }

    /**
     * Delete the last element of the circular queue.
     *
     * @return the deleted last element
     */
    public E removeLast() {
        if (size == 0) {
            throw new NoSuchElementException("The circular queue is empty thus not able to remove the last element");
        }
        final int k = (start + size - 1) % capacity;
        E e = data[k];
        data[k] = null;
        size--;
        return e;
    }

    /**
     * Reset the capacity of the circular queue to newSize. If newSize is smaller than the current
     * data size, only the newSize latest data will be kept. The complexity of this operation is
     * high, which is mainly caused by creating a new circular queue and copying the old data.
     *
     * @param newSize the new maximum capacity of the circular queue
     */
    public void resetCapacity(int newSize) {
        if (newSize == this.capacity) {
            return;
        }
        E[] temp = (E[])new Object[newSize];
        for (int i = Math.max(size - newSize, 0), j = 0; i < size; i++, j++) {
            temp[j] = get(i);
        }
        this.data = temp;
        this.start = 0;
        this.size = Math.min(this.size, newSize);
        this.capacity = newSize;
    }

    /**
     * Get the size of the circular queue.
     *
     * @return the size of the circular queue
     */
    public int size() {
        return size;
    }

    /**
     * Get the maximum capacity of the circular queue.
     *
     * @return the maximum capacity of the circular queue
     */
    public int getCapacity() {
        return capacity;
    }

    /**
     * Return if the circular queue is at full capacity.
     *
     * @return true if the circular queue is at full capacity, otherwise false
     */
    public boolean isAtFullCapacity() {
        return size == capacity;
    }

    /**
     * Clear the circular queue.
     */
    public void clear() {
        if (data != null) {
            Arrays.fill(data, null);
        }
        start = 0;
        size = 0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        CircularQueue<?> that = (CircularQueue<?>)o;
        return start == that.start && size == that.size && capacity == that.capacity && Arrays.deepEquals(data,
            that.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(start, size, capacity);
        result = 31 * result + Arrays.deepHashCode(data);
        return result;
    }

}
