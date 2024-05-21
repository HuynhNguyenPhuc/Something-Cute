class Sort:
    @staticmethod
    def quick_sort(arr):
        if not arr:
            return []
        
        pivot = arr[0]
        less, greater = [], []

        for i in arr[1:]:
            if i < pivot:
                less.append(i)
            elif i >= pivot:
                greater.append(i)
        return Sort.quick_sort(less) + [pivot] + Sort.quick_sort(greater)

    @staticmethod
    def counting_sort(arr):
        if not arr:
            return []
        
        min_value = arr[0]
        max_value = arr[0]
        for value in arr[1:]:
            if value < min_value:
                min_value = value
            elif value > max_value:
                max_value = value
        
        freq = [0] * (max_value - min_value + 1)
        for value in arr:
            freq[value - min_value] += 1

        result = []
        for i in range(0, max_value - min_value + 1):
            result += [i + min_value] * freq[i]

        return result



        
        
