function post_greeting(num) {
  $.ajax({
    type: 'POST',
    url: '/greeting_post',
    // data: JSON.stringify({"key":$('input').data('post-id')}),
    data: JSON.stringify({"key":num}),
    contentType: 'application/json',
  })
  .done(function (data) {
    const greeting = JSON.parse(data.ResultSet).greeting
    document.getElementById('greeting').innerHTML = greeting
  })
  .fail(function() {
    alert('Ajaxに失敗しました');
  })
  // console.log($('input').data('post'))
}